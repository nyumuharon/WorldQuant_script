import os
import sys
import json
import time
import random
import re
import ast
import pandas as pd

# Change working directory to the script's directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Add ACE API folder to path
sys.path.append(r"C:\Users\HP\Downloads\ACE API [Gold]")

# Import WorldQuant helper library functions
import ace_lib
from ace_lib import start_session, simulate_alpha_list, get_operators, get_datafields
from helpful_functions import prettify_result

# Patch get_credentials to automatically authenticate with this specific account
ace_lib.get_credentials = lambda: ("YOUR_EMAIL_HERE", "YOUR_PASSWORD_HERE")

# Monkey-patch start_simulation to automatically retry when 429 Concurrent Limit Exceeded occurs
original_start_simulation = ace_lib.start_simulation
def start_simulation_with_retry(s, simulate_data):
    for attempt in range(12):  # Try for up to 60 seconds (12 * 5s)
        response = original_start_simulation(s, simulate_data)
        if response.status_code == 429:
            print("--> [429 Warning] Concurrent simulation limit reached on server. Waiting 5 seconds to retry...")
            time.sleep(5)
            continue
        return response
    return response
ace_lib.start_simulation = start_simulation_with_retry

class CustomAlphaMutator:
    """
    GP 2.0 Advanced Bug-Free Mutator:
    1. Single Token Mutation: Mutates exactly one token at a time to prevent "cascade mutations."
    2. Arity Safety: Groups and swaps operators strictly by arity/signature.
    3. Active Analyst Operators: Completely excludes delay, sigmoid, and ts_entropy.
    4. Parenthesis-Depth Crossover: Splits formulas only at depth 0.
    """
    def __init__(self, allowed_ops, fields_df):
        # 1. Group active allowed operators strictly by signature
        # Booster (Stateless)
        all_booster_nlb = [("rank", False), ("scale", False)]
        self.booster_no_lookback = [op for op in all_booster_nlb if op[0] in allowed_ops]
        
        # Booster (Lookback)
        all_booster_lb = [("ts_rank", True)]
        self.booster_lookback = [op for op in all_booster_lb if op[0] in allowed_ops]
        
        # Supporting (Stateless)
        all_supp_nlb = [("sign", False), ("ts_backfill", False)]
        self.supporting_no_lookback = [op for op in all_supp_nlb if op[0] in allowed_ops]
        
        # Supporting (Lookback) - ts_delay is used in place of deprecated delay
        all_supp_lb = [
            ("ts_mean", True),
            ("ts_delay", True) if "ts_delay" in allowed_ops else ("delay", True),
            ("ts_arg_max", True),
            ("ts_arg_min", True)
        ]
        self.supporting_lookback = [op for op in all_supp_lb if op[0] in allowed_ops]

        self.lookbacks = [63, 126, 252, 504]

        # Filter fields: ONLY keep 'MATRIX' type fields!
        matrix_df = fields_df[fields_df['type'] == 'MATRIX'].copy()
        matrix_df['alphaCount'] = pd.to_numeric(matrix_df['alphaCount'], errors='coerce').fillna(0)
        matrix_df['weight'] = matrix_df['alphaCount'] + 1.0
        
        # Build category map and weights
        self.supporting_fields_by_cat = {}
        self.supporting_weights_by_cat = {}
        self.field_to_category = {}
        self.field_to_weight = {}
        
        for _, row in matrix_df.iterrows():
            fid = row['id']
            weight = float(row['weight'])
            cat_raw = row['category']
            cat_name = "Other"
            if pd.notna(cat_raw):
                try:
                    cat_name = ast.literal_eval(cat_raw)['name']
                except Exception:
                    try:
                        cat_name = json.loads(cat_raw.replace("'", '"'))['name']
                    except Exception:
                        cat_name = "Other"
            if cat_name not in self.supporting_fields_by_cat:
                self.supporting_fields_by_cat[cat_name] = []
                self.supporting_weights_by_cat[cat_name] = []
            
            self.supporting_fields_by_cat[cat_name].append(fid)
            self.supporting_weights_by_cat[cat_name].append(weight)
            self.field_to_category[fid] = cat_name
            self.field_to_weight[fid] = weight
            
        self.all_fields = list(self.field_to_category.keys())
        
        # Booster fields (Price/Volume category)
        self.booster_fields = self.supporting_fields_by_cat.get("Price Volume", ["close", "open", "volume", "high", "low"])
        self.booster_fields = [f for f in self.booster_fields if f in self.all_fields]
        if not self.booster_fields:
            self.booster_fields = ["close", "open", "volume"]

    def select_weighted(self, choices, weights):
        return random.choices(choices, weights=weights, k=1)[0]

    def mutate_fields(self, formula):
        """Mutate exactly ONE field token in the formula (fixes cascade mutations)."""
        # Find which of all_fields are actually present in the formula
        present_fields = [f for f in self.all_fields if re.search(rf"\b{f}\b", formula)]
        if not present_fields:
            return formula
            
        target_field = random.choice(present_fields)
        cat = self.field_to_category[target_field]
        
        # Select target replacement list (check if it's booster field or supporting field)
        if target_field in self.booster_fields and random.random() < 0.5:
            choices = [f for f in self.booster_fields if f != target_field]
        else:
            choices = [f for f in self.supporting_fields_by_cat[cat] if f != target_field]
            
        if not choices:
            choices = [f for f in self.all_fields if f != target_field]
            
        weights = [self.field_to_weight[f] for f in choices]
        new_field = self.select_weighted(choices, weights)
        
        # Replace exactly ONE occurrence
        return re.sub(rf"\b{target_field}\b", new_field, formula, count=1)

    def mutate_operators(self, formula):
        """Mutate exactly ONE operator token, preserving its lookback signature (arity safety)."""
        # Find which operators are present in the formula
        present_booster_lb = [op for op, _ in self.booster_lookback if op + "(" in formula]
        present_booster_nlb = [op for op, _ in self.booster_no_lookback if op + "(" in formula]
        present_supp_lb = [op for op, _ in self.supporting_lookback if op + "(" in formula]
        present_supp_nlb = [op for op, _ in self.supporting_no_lookback if op + "(" in formula]
        
        categories = []
        if present_booster_lb: categories.append(("booster_lb", present_booster_lb))
        if present_booster_nlb: categories.append(("booster_nlb", present_booster_nlb))
        if present_supp_lb: categories.append(("supp_lb", present_supp_lb))
        if present_supp_nlb: categories.append(("supp_nlb", present_supp_nlb))
        
        if not categories:
            return formula
            
        cat_name, ops_list = random.choice(categories)
        target_op = random.choice(ops_list)
        
        # Swap strictly within the same signature class (arity safety)
        if cat_name == "booster_lb":
            choices = [op[0] for op in self.booster_lookback if op[0] != target_op]
        elif cat_name == "booster_nlb":
            choices = [op[0] for op in self.booster_no_lookback if op[0] != target_op]
        elif cat_name == "supp_lb":
            choices = [op[0] for op in self.supporting_lookback if op[0] != target_op]
        else:
            choices = [op[0] for op in self.supporting_no_lookback if op[0] != target_op]
            
        if choices:
            new_op = random.choice(choices)
            return re.sub(rf"\b{target_op}\(", f"{new_op}(", formula, count=1)
        return formula

    def mutate_lookbacks(self, formula):
        """Mutate exactly ONE lookback parameter in the formula."""
        present_lookbacks = []
        for val in self.lookbacks:
            pattern = rf",\s*{val}\)"
            if re.search(pattern, formula):
                present_lookbacks.append(val)
        
        if not present_lookbacks:
            return formula
            
        target_lookback = random.choice(present_lookbacks)
        choices = [v for v in self.lookbacks if v != target_lookback]
        new_val = random.choice(choices)
        
        pattern = rf",\s*{target_lookback}\)"
        replacement = f", {new_val})"
        return re.sub(pattern, replacement, formula, count=1)

    def mutate_structure(self, formula):
        """GP 2.0 Structural Mutation: Grow (adds operator wrap) or Shrink (prunes operator)."""
        mutated = formula
        # Grow: Wrap a raw field in a new operator
        if random.random() < 0.7:
            present_fields = [f for f in self.all_fields if re.search(rf"\b{f}\b", formula)]
            if present_fields:
                field_to_wrap = random.choice(present_fields)
                op_type = random.choice(["booster", "supporting"])
                if op_type == "booster":
                    op_choices = self.booster_lookback + self.booster_no_lookback
                else:
                    op_choices = self.supporting_lookback + self.supporting_no_lookback
                
                if op_choices:
                    op_name, has_lookback = random.choice(op_choices)
                    if has_lookback:
                        lookback = random.choice(self.lookbacks)
                        replacement = f"{op_name}({field_to_wrap}, {lookback})"
                    else:
                        replacement = f"{op_name}({field_to_wrap})"
                    mutated = re.sub(rf"\b{field_to_wrap}\b", replacement, mutated, count=1)
        # Shrink: Unwrap an operator (replaces 'op(field)' with 'field')
        else:
            all_ops_names = [op[0] for op in (self.booster_lookback + self.booster_no_lookback + self.supporting_lookback + self.supporting_no_lookback)]
            for op in all_ops_names:
                pattern = rf"\b{op}\(([^,)]+)(?:,\s*\d+)?\)"
                match = re.search(pattern, formula)
                if match:
                    field_inside = match.group(1).strip()
                    mutated = re.sub(pattern, field_inside, mutated, count=1)
                    break
        return mutated

    def mutate(self, formula):
        mutation_type = random.choice(["field", "operator", "lookback", "structure"])
        if mutation_type == "field":
            return self.mutate_fields(formula)
        elif mutation_type == "operator":
            return self.mutate_operators(formula)
        elif mutation_type == "lookback":
            return self.mutate_lookbacks(formula)
        else:
            return self.mutate_structure(formula)

    @staticmethod
    def find_top_level_splits(formula):
        """Find indices of operators (+, -, *, /) that occur at parenthesis depth 0."""
        depth = 0
        splits = []
        for i, char in enumerate(formula):
            if char == '(':
                depth += 1
            elif char == ')':
                depth -= 1
            elif char in ('*', '+', '-', '/') and depth == 0:
                splits.append((i, char))
        return splits

    @staticmethod
    def crossover(formula1, formula2):
        """GP 2.0 Crossover: Splits parent formulas only at depth 0 to maintain syntactical validity."""
        splits1 = CustomAlphaMutator.find_top_level_splits(formula1)
        splits2 = CustomAlphaMutator.find_top_level_splits(formula2)
        
        if not splits1 or not splits2:
            # Fallback wrapper if no top-level split is possible
            op = random.choice(["*", "+", "-", "/"])
            return f"({formula1}) {op} ({formula2})", f"({formula2}) {op} ({formula1})"
            
        idx1, op1 = random.choice(splits1)
        idx2, op2 = random.choice(splits2)
        
        part1_left = formula1[:idx1].strip()
        part1_right = formula1[idx1+1:].strip()
        
        part2_left = formula2[:idx2].strip()
        part2_right = formula2[idx2+1:].strip()
        
        new_op1 = random.choice(["*", "+", "-", "/"])
        new_op2 = random.choice(["*", "+", "-", "/"])
        
        child1 = f"{part1_left} {new_op1} {part2_right}"
        child2 = f"{part2_left} {new_op2} {part1_right}"
        return child1, child2

class WQOnlineGP:
    def __init__(self, population_size=12):
        self.population_size = population_size
        self.mutator = None
        self.history = []
        self.passed_alphas = []

        self.sim_config = {
            'get_pnl': False,
            'get_stats': False,
            'save_pnl_file': False,
            'save_stats_file': False,
            'save_result_file': False,
            'check_submission': True,
            'check_self_corr': True,
            'check_prod_corr': True
        }

    def build_payload(self, formula):
        from ace_lib import generate_alpha
        return generate_alpha(
            regular=formula,
            region="USA",
            universe="TOP3000",
            delay=1,
            decay=0, # Decay is strictly 0
            neutralization="SUBINDUSTRY"
        )

    def parse_is_checks(self, is_tests_df):
        checks_data = {
            "sharpe_test": "FAIL",
            "fitness_test": "FAIL",
            "turnover_min_test": "FAIL",
            "turnover_max_test": "FAIL",
            "weight_test": "FAIL",
            "sub_sharpe_test": "FAIL",
            "challenge_test": "FAIL"
        }
        
        if is_tests_df is not None and not is_tests_df.empty:
            for idx, row in is_tests_df.iterrows():
                name = str(row.get("name", "")).upper()
                res = str(row.get("result", "")).upper()
                
                if name in ("LOW_SHARPE", "SHARPE"):
                    checks_data["sharpe_test"] = res
                elif name in ("LOW_FITNESS", "FITNESS"):
                    checks_data["fitness_test"] = res
                elif name in ("LOW_TURNOVER", "TURNOVER_MIN"):
                    checks_data["turnover_min_test"] = res
                elif name in ("HIGH_TURNOVER", "TURNOVER_MAX"):
                    checks_data["turnover_max_test"] = res
                elif name in ("CONCENTRATED_WEIGHT", "WEIGHT"):
                    checks_data["weight_test"] = res
                elif name in ("LOW_SUB_UNIVERSE_SHARPE", "SUB_UNIVERSE_SHARPE"):
                    checks_data["sub_sharpe_test"] = res
                elif name in ("MATCHES_COMPETITION", "COMPETITION", "CHALLENGE"):
                    checks_data["challenge_test"] = res

        return checks_data

    def check_if_passed(self, checks_data):
        return all(status == "PASS" for status in checks_data.values())

    def get_passed_count(self, checks_data):
        return sum(1 for status in checks_data.values() if status == "PASS")

    def select_parent_tournament(self, tournament_size=3):
        """Tournament Selection: Pick best of 3 random formulas from history."""
        candidates = random.sample(self.history, min(len(self.history), tournament_size))
        candidates = sorted(candidates, key=lambda x: (x.get("passed_count", 0), x.get("fitness", 0)), reverse=True)
        return candidates[0]["formula"]

    def process_simulation_results(self, results, generation, origin_status):
        for item in results:
            formula = item["simulate_data"]["regular"]
            alpha_id = item.get("alpha_id")
            sharpe, fitness, turnover, margin = 0.0, 0.0, 0.0, 0.0
            
            is_stats = item.get("is_stats")
            if is_stats is not None and not is_stats.empty:
                sharpe = float(is_stats.iloc[0].get("sharpe", 0.0))
                fitness = float(is_stats.iloc[0].get("fitness", 0.0))
                turnover = float(is_stats.iloc[0].get("turnover", 0.0))
                margin = float(is_stats.iloc[0].get("margin", 0.0))

            is_tests = item.get("is_tests")
            checks_data = self.parse_is_checks(is_tests)
            
            passed_all = self.check_if_passed(checks_data)
            passed_status = "PASS" if passed_all else "FAIL"
            passed_count = self.get_passed_count(checks_data)

            record = {
                "generation": generation,
                "formula": formula,
                "alpha_id": alpha_id,
                "sharpe": sharpe,
                "fitness": fitness,
                "turnover": turnover,
                "margin_or_sub_sharpe": margin,
                "status": passed_status,
                "passed_count": passed_count,
                "origin": origin_status
            }
            record.update(checks_data)
            self.history.append(record)

            if passed_all:
                passed_record = {
                    "formula": formula,
                    "alpha_id": alpha_id,
                    "sharpe": sharpe,
                    "fitness": fitness,
                    "turnover": turnover,
                    "margin_or_sub_sharpe": margin
                }
                passed_record.update(checks_data)
                
                if not any(x["formula"] == formula for x in self.passed_alphas):
                    self.passed_alphas.append(passed_record)
                    print(f"--> [PASSED] Alpha {alpha_id} passed all 7 checks! Added to golden list.")

            yield {
                "formula": formula,
                "sharpe": sharpe,
                "fitness": fitness,
                "passed_count": passed_count
            }

    def run_online_evolution(self, seeds):
        print(f"\n==============================================")
        print(f"STARTING GOAL-DIRECTED GP 2.0 ONLINE SEARCH")
        print(f"Population size: {self.population_size}")
        print(f"To stop the script at any time, press Ctrl+C. Your progress is saved after each generation.")
        print(f"==============================================\n")

        s = start_session()
        print("Session established successfully.")

        print("Querying allowed operators from WorldQuant Brain...")
        ops_df = get_operators(s)
        allowed_ops = set(ops_df['name'].tolist())

        print("Querying allowed datafields for USA region...")
        fields_df = get_datafields(s, region="USA", delay=1, universe="TOP3000")

        self.mutator = CustomAlphaMutator(allowed_ops, fields_df)
        print(f"Successfully loaded {len(allowed_ops)} allowed operators and {len(self.mutator.all_fields)} allowed MATRIX fields.")

        # Load past seeds
        passed_path = "passed_alphas.csv"
        if os.path.exists(passed_path):
            try:
                passed_df = pd.read_csv(passed_path)
                if not passed_df.empty and "formula" in passed_df.columns:
                    loaded_formulas = passed_df["formula"].dropna().tolist()
                    if loaded_formulas:
                        print(f"Loaded {len(loaded_formulas)} successful 7/7 seeds from past runs. Using them as starting seeds!")
                        seeds = (loaded_formulas + seeds)[:max(len(loaded_formulas), len(seeds))]
            except Exception as e:
                print(f"Note: Could not load past seeds from file: {e}")

        cleaned_seeds = []
        for formula in seeds:
            cleaned = formula
            if "ts_delay" in allowed_ops:
                cleaned = re.sub(r"\bdelay\(", "ts_delay(", cleaned)
            cleaned_seeds.append(cleaned)

        print(f"Backtesting {len(cleaned_seeds)} seed formulas...")
        seed_results = simulate_alpha_list(
            s, 
            [self.build_payload(f) for f in cleaned_seeds], 
            limit_of_concurrent_simulations=3, 
            simulation_config=self.sim_config
        )
        current_population = list(self.process_simulation_results(seed_results, 0, "seed"))
        self.save_results_to_disk()

        gen = 1
        while True:
            try:
                print(f"\n======================================")
                print(f"EVOLVING GENERATION {gen} (Endless Mode)...")
                print(f"\n======================================")

                offspring_formulas = []
                population_formulas = [item["formula"] for item in current_population]

                while len(offspring_formulas) < self.population_size:
                    if random.random() < 0.8:
                        parent = self.select_parent_tournament()
                        child = self.mutator.mutate(parent)
                        if child not in population_formulas and child not in offspring_formulas:
                            offspring_formulas.append(child)
                    else:
                        parent1 = self.select_parent_tournament()
                        parent2 = self.select_parent_tournament()
                        if parent1 != parent2:
                            child1, child2 = CustomAlphaMutator.crossover(parent1, parent2)
                            for child in (child1, child2):
                                if len(offspring_formulas) < self.population_size and child not in population_formulas and child not in offspring_formulas:
                                    offspring_formulas.append(child)

                print(f"Submitting {len(offspring_formulas)} offspring formulas to simulator...")
                offspring_payloads = [self.build_payload(f) for f in offspring_formulas]
                offspring_results = simulate_alpha_list(
                    s, 
                    offspring_payloads, 
                    limit_of_concurrent_simulations=3, 
                    simulation_config=self.sim_config
                )
                new_candidates = list(self.process_simulation_results(offspring_results, gen, "evolved"))

                combined_pool = current_population + new_candidates
                combined_pool = sorted(combined_pool, key=lambda x: (x.get("passed_count", 0), x.get("fitness", 0)), reverse=True)
                current_population = combined_pool[:self.population_size]

                self.save_results_to_disk()
                print(f"Generation {gen} saved successfully. Total passed alphas so far: {len(self.passed_alphas)}")
                
                gen += 1
                
            except KeyboardInterrupt:
                print("\nKeyboardInterrupt detected. Shutting down evolution loop gracefully...")
                break
            except Exception as e:
                print(f"\nError during generation {gen}: {e}. Retrying in 10 seconds...")
                time.sleep(10)

    def save_results_to_disk(self):
        history_df = pd.DataFrame(self.history)
        history_df.to_csv("gp_live_search_results.csv", index=False)
        
        passed_df = pd.DataFrame(self.passed_alphas)
        if passed_df.empty:
            passed_df = pd.DataFrame(columns=["formula", "alpha_id", "sharpe", "fitness", "turnover", "margin_or_sub_sharpe", "sharpe_test", "fitness_test", "turnover_min_test", "turnover_max_test", "weight_test", "sub_sharpe_test", "challenge_test"])
        passed_df.to_csv("passed_alphas.csv", index=False)

if __name__ == "__main__":
    seed_formulas = [
        "ts_mean(rank(close), 252) * rank(volume)",
        "ts_mean(log(open / close), 126) * scale(volume)",
        "ts_delay(rank(close), 63) * ts_rank(volume, 126)"
    ]
    gp = WQOnlineGP(population_size=12)
    gp.run_online_evolution(seed_formulas)
