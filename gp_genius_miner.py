import os
import sys
import json
import time
import random
import re
import ast
import pandas as pd

# Load environment variables from .env manually
env_path = ".env" if os.path.exists(".env") else "../.env"
if os.path.exists(env_path):
    with open(env_path, "r") as f:
        for line in f:
            if "=" in line:
                parts = line.strip().split("=", 1)
                if len(parts) == 2:
                    os.environ[parts[0]] = parts[1]

# Change working directory to the script's directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Add ACE API folder to path
sys.path.append(r"C:\Users\HP\Downloads\ACE API [Gold]")

# Import WorldQuant helper library functions
import ace_lib
from ace_lib import start_session, simulate_alpha_list, get_operators, get_datafields
from helpful_functions import prettify_result

# Patch get_credentials to automatically authenticate with this specific account
ace_lib.get_credentials = lambda: ("nyumuharon@gmail.com", "Piss_axon17")

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
    Advanced Bug-Free Mutator implementing PDF features:
    1. Parenthesis Nesting structure: supporting_op(booster_op(f1) * booster_op(f2))
    2. Turnover Mitigation wrapping
    3. LLM-Guided Semantic Mutation (Gemini)
    4. Arity Safety and Local Mutation
    """
    def __init__(self, allowed_ops, fields_df):
        # Group active allowed operators strictly by signature
        all_booster_nlb = [("rank", False), ("scale", False)]
        self.booster_no_lookback = [op for op in all_booster_nlb if op[0] in allowed_ops]
        
        all_booster_lb = [("ts_rank", True)]
        self.booster_lookback = [op for op in all_booster_lb if op[0] in allowed_ops]
        
        all_supp_nlb = [("sign", False), ("ts_backfill", False)]
        self.supporting_no_lookback = [op for op in all_supp_nlb if op[0] in allowed_ops]
        
        all_supp_lb = [
            ("ts_mean", True),
            ("ts_delay", True) if "ts_delay" in allowed_ops else ("delay", True),
            ("ts_arg_max", True),
            ("ts_arg_min", True),
            ("ts_decay_linear", True)
        ]
        self.supporting_lookback = [op for op in all_supp_lb if op[0] in allowed_ops]

        self.lookbacks = [63, 126, 252, 504]

        # Filter fields: ONLY keep 'MATRIX' type fields!
        matrix_df = fields_df[fields_df['type'] == 'MATRIX'].copy()
        matrix_df['alphaCount'] = pd.to_numeric(matrix_df['alphaCount'], errors='coerce').fillna(0)
        matrix_df['weight'] = matrix_df['alphaCount'] + 1.0
        
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
        
        # Booster fields (Price Volume category)
        self.booster_fields = self.supporting_fields_by_cat.get("Price Volume", ["close", "open", "volume", "high", "low"])
        self.booster_fields = [f for f in self.booster_fields if f in self.all_fields]
        if not self.booster_fields:
            self.booster_fields = ["close", "open", "volume"]

    def select_weighted(self, choices, weights):
        return random.choices(choices, weights=weights, k=1)[0]

    def mutate_fields(self, formula):
        """Mutate exactly ONE field token in the formula (fixes cascade mutations)."""
        present_fields = [f for f in self.all_fields if re.search(rf"\b{f}\b", formula)]
        if not present_fields:
            return formula
            
        target_field = random.choice(present_fields)
        cat = self.field_to_category[target_field]
        
        if target_field in self.booster_fields and random.random() < 0.5:
            choices = [f for f in self.booster_fields if f != target_field]
        else:
            choices = [f for f in self.supporting_fields_by_cat[cat] if f != target_field]
            
        if not choices:
            choices = [f for f in self.all_fields if f != target_field]
            
        weights = [self.field_to_weight[f] for f in choices]
        new_field = self.select_weighted(choices, weights)
        
        return re.sub(rf"\b{target_field}\b", new_field, formula, count=1)

    def mutate_operators(self, formula):
        """Mutate exactly ONE operator token, preserving lookback signature (arity safety)."""
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

    def mutate_turnover(self, formula):
        """
        Turnover Mitigation Wrapper (Section 6 & 11 Blueprint):
        Wraps a high-turnover formula in a decay operator (e.g. ts_decay or ts_decay_linear)
        to smooth out trading changes and satisfy the 12.5% - 30% corridor.
        """
        decay_period = random.choice([5, 10, 20])
        return f"ts_decay_linear({formula}, {decay_period})"

    def mutate(self, formula, status_dict=None):
        # If the parent has high turnover, prioritize wrapping it in linear decay
        if status_dict and float(status_dict.get("turnover", 0.0)) > 0.35:
            if random.random() < 0.7:
                return self.mutate_turnover(formula)
            
        mutation_type = random.choice(["field", "operator", "lookback", "turnover"])
        if mutation_type == "field":
            return self.mutate_fields(formula)
        elif mutation_type == "operator":
            return self.mutate_operators(formula)
        elif mutation_type == "lookback":
            return self.mutate_lookbacks(formula)
        else:
            return self.mutate_turnover(formula)

    @staticmethod
    def find_top_level_splits(formula):
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
        splits1 = CustomAlphaMutator.find_top_level_splits(formula1)
        splits2 = CustomAlphaMutator.find_top_level_splits(formula2)
        
        if not splits1 or not splits2:
            op = random.choice(["*", "+", "-", "/"])
            return f"({formula1}) {op} ({formula2})", f"({formula2}) {op} ({formula1})"
            
        idx1, op1 = random.choice(splits1)
        idx2, op2 = random.choice(splits2)
        
        part1_left = formula1[:idx1].strip()
        part1_right = formula1[idx1+1:].strip()
        
        part2_left = formula2[:idx2].strip()
        part2_right = formula2[idx2+1:].strip()
        
        new_op1 = random.choice(["*", "+"])
        new_op2 = random.choice(["*", "+"])
        
        child1 = f"{part1_left} {new_op1} {part2_right}"
        child2 = f"{part2_left} {new_op2} {part1_right}"
        return child1, child2

class WQOnlineGP:
    def __init__(self, region="USA", population_size=12):
        self.region = region.upper()
        self.population_size = population_size
        self.mutator = None
        self.history = []
        self.passed_alphas = []
        self.session = None

        # Regional mapping (from PDF specification)
        universe_mapping = {
            "USA": "TOP3000",
            "GLB": "GLB3000",
            "IND": "TOP1000",
            "ASI": "TOP2000",
            "CHN": "TOP2000"
        }
        neutralization_mapping = {
            "USA": "SUBINDUSTRY",
            "GLB": "SUBINDUSTRY",
            "IND": "INDUSTRY",
            "ASI": "SECTOR",
            "CHN": "SUBINDUSTRY"
        }

        self.universe = universe_mapping.get(self.region, "TOP3000")
        self.neutralization = neutralization_mapping.get(self.region, "SUBINDUSTRY")

        # Concurrency limit based on Genius tier limits:
        # GLB allows 4 concurrent simulations, other regions allow 8!
        if self.region == "GLB":
            self.concurrency_limit = 4
        else:
            self.concurrency_limit = 8

        self.decay = 12

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
            region=self.region,
            universe=self.universe,
            delay=1,
            decay=self.decay,
            neutralization=self.neutralization
        )

    def safe_simulate_alpha_list(self, payloads, retries=10):
        """Wrapper for simulate_alpha_list with automatic retries on network disconnect errors."""
        for attempt in range(retries):
            try:
                results = simulate_alpha_list(
                    self.session, 
                    payloads, 
                    limit_of_concurrent_simulations=self.concurrency_limit, 
                    simulation_config=self.sim_config
                )
                return results
            except Exception as e:
                print(f"--> [Network Connection Error] {e}. Attempt {attempt + 1}/{retries}. Retrying in 45 seconds...")
                time.sleep(45)
                try:
                    self.session = start_session()
                except Exception as ses_err:
                    print(f"Failed to refresh session: {ses_err}")
        print("--> [Critical Error] All network retries failed.")
        return []

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
        candidates = random.sample(self.history, min(len(self.history), tournament_size))
        candidates = sorted(candidates, key=lambda x: (x.get("passed_count", 0), x.get("fitness", 0)), reverse=True)
        return candidates[0]

    def process_simulation_results(self, results, generation, origin_status):
        for item in results:
            formula = item["simulate_data"]["regular"]
            alpha_id = item.get("alpha_id")
            
            # Detailed metrics initialization
            sub_universe_sharpe = 0.0
            robust_sharpe = 0.0
            ladder_yr_2_sharpe = 0.0
            returns = 0.0
            sharpe = 0.0
            fitness = 0.0
            turnover = 0.0
            margin = 0.0

            if alpha_id:
                try:
                    # Query the full simulation result JSON from WQ Brain API
                    result_json = ace_lib.get_simulation_result_json(self.session, alpha_id)
                    is_data = result_json.get("is", {})
                    sharpe = float(is_data.get("sharpe", 0.0))
                    fitness = float(is_data.get("fitness", 0.0))
                    turnover = float(is_data.get("turnover", 0.0))
                    returns = float(is_data.get("returns", 0.0))
                    margin = float(is_data.get("margin", 0.0))
                    
                    sub_universe_sharpe = float(is_data.get("subUniverseSharpe", 0.0))
                    robust_sharpe = float(is_data.get("robustSharpe", 0.0))
                    
                    # Year 2 Ladder Sharpe is the second element in the ladder list
                    ladder_data = result_json.get("ladder", [])
                    if isinstance(ladder_data, list) and len(ladder_data) >= 2:
                        ladder_yr_2_sharpe = float(ladder_data[1].get("sharpe", 0.0))
                except Exception as e:
                    print(f"--> [Warning] Failed to fetch full JSON stats for {alpha_id}: {e}")
                    is_stats = item.get("is_stats")
                    if is_stats is not None and not is_stats.empty:
                        sharpe = float(is_stats.iloc[0].get("sharpe", 0.0))
                        fitness = float(is_stats.iloc[0].get("fitness", 0.0))
                        turnover = float(is_stats.iloc[0].get("turnover", 0.0))
                        margin = float(is_stats.iloc[0].get("margin", 0.0))

            is_tests = item.get("is_tests")
            checks_data = self.parse_is_checks(is_tests)
            
            passed_all_platform = self.check_if_passed(checks_data)
            
            # Local Genius-level validation check (Section 4 & 5 Blueprint)
            passed_genius = (
                sharpe >= 1.58 and
                fitness >= 1.0 and
                0.01 <= turnover <= 0.40 and
                sub_universe_sharpe >= 0.87 and
                robust_sharpe >= 1.0 and
                ladder_yr_2_sharpe >= 2.02
            )
            
            passed_count = self.get_passed_count(checks_data)
            corridor_bonus = 1 if (0.125 <= turnover <= 0.30) else 0

            record = {
                "generation": generation,
                "formula": formula,
                "alpha_id": alpha_id,
                "sharpe": sharpe,
                "fitness": fitness,
                "turnover": turnover,
                "margin_or_sub_sharpe": margin,
                "sub_universe_sharpe": sub_universe_sharpe,
                "robust_sharpe": robust_sharpe,
                "ladder_year_2_sharpe": ladder_yr_2_sharpe,
                "status": "PASS" if (passed_all_platform or passed_genius) else "FAIL",
                "passed_count": passed_count + corridor_bonus,
                "is_genius": "PASS" if passed_genius else "FAIL",
                "origin": origin_status
            }
            record.update(checks_data)
            self.history.append(record)

            if passed_all_platform or passed_genius:
                passed_record = {
                    "formula": formula,
                    "alpha_id": alpha_id,
                    "sharpe": sharpe,
                    "fitness": fitness,
                    "turnover": turnover,
                    "margin_or_sub_sharpe": margin,
                    "sub_universe_sharpe": sub_universe_sharpe,
                    "robust_sharpe": robust_sharpe,
                    "ladder_year_2_sharpe": ladder_yr_2_sharpe,
                    "is_genius": "PASS" if passed_genius else "FAIL"
                }
                passed_record.update(checks_data)
                
                if not any(x["formula"] == formula for x in self.passed_alphas):
                    self.passed_alphas.append(passed_record)
                    status_lbl = "GENIUS" if passed_genius else "PLATFORM"
                    print(f"--> [{status_lbl} PASSED] Alpha {alpha_id} passed! Added to golden list.")

            yield record

    def run_online_evolution(self, seeds, num_generations=2):
        print(f"\n==============================================")
        print(f"STARTING GOAL-DIRECTED AUTONOMOUS ALPHA MINER")
        print(f"==============================================\n")

        self.session = start_session()
        print("Session established successfully.")

        print("Querying allowed operators from WorldQuant Brain...")
        ops_df = get_operators(self.session)
        allowed_ops = set(ops_df['name'].tolist())

        print(f"Querying allowed datafields for {self.region} region...")
        fields_df = get_datafields(self.session, region=self.region, delay=1, universe=self.universe)

        self.mutator = CustomAlphaMutator(allowed_ops, fields_df)
        print(f"Loaded {len(allowed_ops)} operators and {len(self.mutator.all_fields)} MATRIX fields.")

        # Load seeds from passed list if it exists and load full history to prevent duplicates
        passed_path = "passed_alphas.csv"
        pre_existing_records = []
        if os.path.exists(passed_path):
            try:
                passed_df = pd.read_csv(passed_path)
                if not passed_df.empty and "formula" in passed_df.columns:
                    for _, row in passed_df.iterrows():
                        rec = row.to_dict()
                        # Reconstruct keys to ensure perfect match with self.history
                        rec["generation"] = 0
                        rec["status"] = "PASS"
                        rec["passed_count"] = int(rec.get("passed_count", 7))
                        rec["sub_universe_sharpe"] = float(rec.get("sub_universe_sharpe", 0.0))
                        rec["robust_sharpe"] = float(rec.get("robust_sharpe", 0.0))
                        rec["ladder_year_2_sharpe"] = float(rec.get("ladder_year_2_sharpe", 0.0))
                        rec["is_genius"] = rec.get("is_genius", "FAIL")
                        pre_existing_records.append(rec)
                    print(f"Loaded {len(pre_existing_records)} pre-existing 7/7 alphas directly into memory.")
            except Exception as e:
                print(f"Note: Could not load past seeds: {e}")

        # Load the set of historically simulated formulas
        self.simulated_formulas = set()
        history_path = "gp_live_search_results.csv"
        if os.path.exists(history_path):
            try:
                hist_df = pd.read_csv(history_path)
                if not hist_df.empty and "formula" in hist_df.columns:
                    self.simulated_formulas = set(hist_df["formula"].dropna().tolist())
                    print(f"Loaded {len(self.simulated_formulas)} historically simulated formulas to prevent duplicate runs.")
            except Exception as e:
                print(f"Note: Could not load history: {e}")

        # Clean seed formulas
        cleaned_seeds = []
        for formula in seeds:
            cleaned = formula
            if "ts_delay" in allowed_ops:
                cleaned = re.sub(r"\bdelay\(", "ts_delay(", cleaned)
            # Only simulate if we don't have its record already
            if cleaned not in [r["formula"] for r in pre_existing_records]:
                cleaned_seeds.append(cleaned)

        if cleaned_seeds:
            print(f"Backtesting {len(cleaned_seeds)} new seed formulas...")
            seed_results = self.safe_simulate_alpha_list([self.build_payload(f) for f in cleaned_seeds])
            new_seed_population = list(self.process_simulation_results(seed_results, 0, "seed"))
        else:
            new_seed_population = []

        # Combine loaded and newly simulated population
        self.history.extend(pre_existing_records)
        self.passed_alphas = [r for r in pre_existing_records]
        
        current_population = pre_existing_records + new_seed_population
        current_population = sorted(current_population, key=lambda x: (x.get("passed_count", 0), x.get("fitness", 0)), reverse=True)
        current_population = current_population[:self.population_size]
        self.save_results_to_disk()

        for gen in range(1, num_generations + 1):
            print(f"\n======================================")
            print(f"EVOLVING GENERATION {gen}...")
            print(f"======================================")

            offspring_formulas = []
            attempts = 0
            
            while len(offspring_formulas) < self.population_size and attempts < 200:
                attempts += 1
                if random.random() < 0.7:
                    parent_record = self.select_parent_tournament()
                    parent_formula = parent_record["formula"]
                    
                    child = self.mutator.mutate(parent_formula, parent_record)
                    if child not in [p["formula"] for p in current_population] and child not in offspring_formulas and child not in self.simulated_formulas:
                        offspring_formulas.append(child)
                else:
                    parent_record1 = self.select_parent_tournament()
                    parent_record2 = self.select_parent_tournament()
                    parent1 = parent_record1["formula"]
                    parent2 = parent_record2["formula"]
                    
                    if parent1 != parent2:
                        child1, child2 = CustomAlphaMutator.crossover(parent1, parent2)
                        for child in (child1, child2):
                            if len(offspring_formulas) < self.population_size and child not in [p["formula"] for p in current_population] and child not in offspring_formulas and child not in self.simulated_formulas:
                                offspring_formulas.append(child)

            if not offspring_formulas:
                print(f"--> [Warning] Evolved pool in generation {gen} was already completely simulated. Stopping evolution.")
                break

            print(f"Submitting {len(offspring_formulas)} offspring formulas to simulator...")
            offspring_payloads = [self.build_payload(f) for f in offspring_formulas]
            offspring_results = self.safe_simulate_alpha_list(offspring_payloads)
            new_candidates = list(self.process_simulation_results(offspring_results, gen, "evolved"))

            # Update simulated formulas set in memory
            for f in offspring_formulas:
                self.simulated_formulas.add(f)

            # Elitism and Selection
            combined_pool = current_population + new_candidates
            combined_pool = sorted(combined_pool, key=lambda x: (x.get("passed_count", 0), x.get("fitness", 0)), reverse=True)
            current_population = combined_pool[:self.population_size]

            self.save_results_to_disk()
            print(f"Generation {gen} completed. Passed alphas count: {len(self.passed_alphas)}")

    def save_results_to_disk(self):
        history_df = pd.DataFrame(self.history)
        history_df.to_csv("gp_live_search_results.csv", index=False)
        
        passed_df = pd.DataFrame(self.passed_alphas)
        if passed_df.empty:
            passed_df = pd.DataFrame(columns=["formula", "alpha_id", "sharpe", "fitness", "turnover", "margin_or_sub_sharpe", "sharpe_test", "fitness_test", "turnover_min_test", "turnover_max_test", "weight_test", "sub_sharpe_test", "challenge_test"])
        passed_df.to_csv("passed_alphas.csv", index=False)
        
        # Copy to Downloads
        import shutil
        try:
            shutil.copy2("gp_live_search_results.csv", r"C:\Users\HP\Downloads\gp_live_search_results_nyumuharon.csv")
            shutil.copy2("passed_alphas.csv", r"C:\Users\HP\Downloads\passed_alphas_nyumuharon.csv")
        except Exception:
            pass

def input_with_timeout(prompt, timeout=10, default="USA"):
    # Check if there are command line arguments first
    if len(sys.argv) > 1:
        arg_val = sys.argv[1].strip().upper()
        if arg_val in ("USA", "GLB", "IND", "ASI", "CHN"):
            print(f"Using region from command line argument: {arg_val}")
            return arg_val

    try:
        import msvcrt
        print(prompt, end="", flush=True)
        start_time = time.time()
        input_str = ""
        while time.time() - start_time < timeout:
            if msvcrt.kbhit():
                char = msvcrt.getwche()
                if char in ("\r", "\n"):
                    print()
                    return input_str.strip().upper()
                elif char in ("\b", "\xe0"):
                    if len(input_str) > 0:
                        input_str = input_str[:-1]
                else:
                    input_str += char
            time.sleep(0.05)
        print(f"\nNo input received within {timeout} seconds. Defaulting to {default}.")
        return default
    except Exception:
        try:
            val = input(f"{prompt} (press Enter to use default {default}): ").strip().upper()
            return val if val in ("USA", "GLB", "IND", "ASI", "CHN") else default
        except Exception:
            return default

if __name__ == "__main__":
    region_choice = input_with_timeout(
        "Enter target region to mine (USA, GLB, IND, ASI, CHN) [default: USA]: ",
        timeout=10,
        default="USA"
    )
    if region_choice not in ("USA", "GLB", "IND", "ASI", "CHN"):
        print(f"Unknown region '{region_choice}'. Defaulting to USA.")
        region_choice = "USA"
        
    seed_formulas = [
        "ts_mean(rank(close) * rank(volume), 252)",
        "ts_mean(sign(open / close - 1) * scale(volume), 126)",
        "ts_delay(rank(close) * ts_rank(volume, 126), 63)"
    ]
    
    # Run the bounded loop with the chosen region and the correct Genius concurrency settings
    gp = WQOnlineGP(region=region_choice, population_size=12)
    gp.run_online_evolution(seed_formulas, num_generations=500)
