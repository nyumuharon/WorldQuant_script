import os
import sys
import json
import time
import random
import re
import ast
import pandas as pd

# Dual-logging system redirecting standard output/error to both terminal and a file
class Logger(object):
    def __init__(self, filename="gp_genius_miner.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding="utf-8")
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
    def flush(self):
        self.terminal.flush()
        self.log.flush()

sys.stdout = Logger()
sys.stderr = Logger()

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
ace_lib.get_credentials = lambda: ("sungura693@gmail.com", "Sofia@123")

# Monkey-patch check_session_and_relogin to completely disable background session refreshes.
# This guarantees that the script will NEVER perform background logins during the night.
ace_lib.check_session_and_relogin = lambda s: s

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

# Monkey-patch simulate_single_alpha to catch network drops inside the thread pool
original_simulate_single_alpha = ace_lib.simulate_single_alpha
def safe_simulate_single_alpha(s, simulate_data):
    try:
        return original_simulate_single_alpha(s, simulate_data)
    except Exception as e:
        print(f"--> [Warning] simulate_single_alpha failed due to connection reset: {e}. Retrying with empty ID.")
        return {"alpha_id": None, "simulate_data": simulate_data}
ace_lib.simulate_single_alpha = safe_simulate_single_alpha

# Monkey-patch get_specified_alpha_stats to catch KeyError 'train'/'test' on failed/timed-out simulations
original_get_specified_alpha_stats = ace_lib.get_specified_alpha_stats
def safe_get_specified_alpha_stats(s, alpha_id, simulate_data, *args, **kwargs):
    try:
        return original_get_specified_alpha_stats(s, alpha_id, simulate_data, *args, **kwargs)
    except Exception as e:
        print(f"--> [Warning] get_specified_alpha_stats encountered error: {e}. Recovering safely.")
        return {
            "alpha_id": alpha_id,
            "simulate_data": simulate_data,
            "is_stats": None,
            "pnl": None,
            "stats": None,
            "is_tests": None,
            "train": None,
            "test": None
        }
ace_lib.get_specified_alpha_stats = safe_get_specified_alpha_stats

class CustomAlphaMutator:
    """
    Advanced Bug-Free Mutator targeting region-specific fields and templates.
    """
    def __init__(self, allowed_ops, regional_fields):
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

        # Filter regional fields: ONLY keep 'MATRIX' type fields!
        self.all_fields = {}
        self.booster_fields = {}
        self.supporting_fields_by_cat = {}
        self.field_to_category = {}
        self.field_to_weight = {}

        for region, fields_df in regional_fields.items():
            matrix_df = fields_df[fields_df['type'] == 'MATRIX'].copy()
            matrix_df['alphaCount'] = pd.to_numeric(matrix_df['alphaCount'], errors='coerce').fillna(0)
            matrix_df['weight'] = matrix_df['alphaCount'] + 1.0

            self.all_fields[region] = []
            self.supporting_fields_by_cat[region] = {}
            
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
                
                if cat_name not in self.supporting_fields_by_cat[region]:
                    self.supporting_fields_by_cat[region][cat_name] = []
                
                self.supporting_fields_by_cat[region][cat_name].append(fid)
                self.all_fields[region].append(fid)
                self.field_to_category[(region, fid)] = cat_name
                self.field_to_weight[(region, fid)] = weight

            # Extract price volume fields for boosters
            self.booster_fields[region] = self.supporting_fields_by_cat[region].get("Price Volume", ["close", "open", "volume", "high", "low"])
            self.booster_fields[region] = [f for f in self.booster_fields[region] if f in self.all_fields[region]]
            if not self.booster_fields[region]:
                self.booster_fields[region] = ["close", "open", "volume"]

    def select_weighted(self, region, choices):
        weights = [self.field_to_weight.get((region, f), 1.0) for f in choices]
        return random.choices(choices, weights=weights, k=1)[0]

    def mutate_fields(self, formula, region):
        """Mutate exactly ONE field token in the formula for the specified region."""
        present_fields = [f for f in self.all_fields[region] if re.search(rf"\b{f}\b", formula)]
        if not present_fields:
            return formula
            
        target_field = random.choice(present_fields)
        cat = self.field_to_category.get((region, target_field), "Other")
        
        if target_field in self.booster_fields[region] and random.random() < 0.5:
            choices = [f for f in self.booster_fields[region] if f != target_field]
        else:
            choices = [f for f in self.supporting_fields_by_cat[region].get(cat, []) if f != target_field]
            
        if not choices:
            choices = [f for f in self.all_fields[region] if f != target_field]
            
        new_field = self.select_weighted(region, choices)
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
        """Wraps a formula (or the final expression in a multi-statement formula) in a linear decay function."""
        decay_period = random.choice([5, 10, 20])
        if ";" in formula:
            statements = [s.strip() for s in formula.split(";") if s.strip()]
            if statements:
                statements[-1] = f"ts_decay_linear({statements[-1]}, {decay_period})"
                return "; ".join(statements) + ";"
        return f"ts_decay_linear({formula}, {decay_period})"

    def mutate(self, formula, region, status_dict=None):
        if status_dict and float(status_dict.get("turnover", 0.0)) > 0.35:
            if random.random() < 0.7:
                child = self.mutate_turnover(formula)
                print(f"[MUTATE] [{region}] Wrapped high-turnover parent: {formula} -> child: {child}")
                return child
            
        mutation_type = random.choice(["field", "operator", "lookback", "turnover"])
        if mutation_type == "field":
            child = self.mutate_fields(formula, region)
        elif mutation_type == "operator":
            child = self.mutate_operators(formula)
        elif mutation_type == "lookback":
            child = self.mutate_lookbacks(formula)
        else:
            child = self.mutate_turnover(formula)
        print(f"[MUTATE] [{region}] Type: {mutation_type} | Parent: {formula} -> Child: {child}")
        return child

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
            child1, child2 = f"({formula1}) {op} ({formula2})", f"({formula2}) {op} ({formula1})"
            print(f"[CROSSOVER] No splits. Bred: {formula1} & {formula2} -> children: {child1} / {child2}")
            return child1, child2
            
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
        print(f"[CROSSOVER] Split bred: {formula1} & {formula2} -> children: {child1} / {child2}")
        return child1, child2

class WQOnlineGP:
    def __init__(self, regions=["USA", "GLB", "IND", "ASI"], population_size_per_region=6):
        self.regions = [r.upper() for r in regions]
        self.population_size_per_region = population_size_per_region
        self.mutator = None
        self.history = []
        self.passed_alphas = []
        self.session = None
        self.last_login_time = time.time()

        # Regional Mappings (optimized for Genius-level passed settings)
        self.universe_mapping = {
            "USA": "TOP3000",
            "GLB": "MINVOL1M",
            "IND": "TOP500",
            "ASI": "MINVOL1M"
        }
        self.neutralization_mapping = {
            "USA": "INDUSTRY",
            "GLB": "STATISTICAL",
            "IND": "STATISTICAL",
            "ASI": "STATISTICAL"
        }
        self.decay_mapping = {
            "USA": 4,
            "GLB": 5,
            "IND": 5,
            "ASI": 7
        }

        # Concurrency limit safe default for mixed batch
        self.concurrency_limit = 4

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

    def build_payload(self, formula, region):
        from ace_lib import generate_alpha
        return generate_alpha(
            regular=formula,
            region=region,
            universe=self.universe_mapping.get(region, "TOP3000"),
            delay=1,
            decay=self.decay_mapping.get(region, 12),
            neutralization=self.neutralization_mapping.get(region, "SUBINDUSTRY")
        )

    def keep_alive_session(self):
        """Sends a quick heartbeat check to the server to keep the session active and prevent expiration."""
        now = time.time()
        if not hasattr(self, "last_heartbeat_time"):
            self.last_heartbeat_time = 0.0
        if now - self.last_heartbeat_time > 300:
            try:
                from ace_lib import check_session_timeout
                check_session_timeout(self.session)
                self.last_heartbeat_time = now
            except Exception:
                pass

    def safe_simulate_alpha_list(self, payloads, retries=1000):
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
        print("--> [Critical Error] All network retries failed.")
        return []

    def safe_query_api(self, api_func, *args, **kwargs):
        """Wrapper to query the WQ Brain API with automatic infinite retries on connection loss."""
        for attempt in range(1000):
            try:
                # Replace the first argument with self.session if it's a SingleSession type
                args_list = list(args)
                if args_list:
                    args_list[0] = self.session
                return api_func(*args_list, **kwargs)
            except Exception as e:
                print(f"--> [Network Error in API Query] {e}. Attempt {attempt + 1}/1000. Retrying query in 15 seconds...")
                time.sleep(15)
        return None

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

    def select_parent_tournament(self, region_population, tournament_size=3):
        candidates = random.sample(region_population, min(len(region_population), tournament_size))
        candidates = sorted(candidates, key=lambda x: (x.get("passed_count", 0), x.get("fitness", 0)), reverse=True)
        return candidates[0]

    def process_simulation_results(self, results, generation, origin_status):
        for item in results:
            formula = item["simulate_data"]["regular"]
            region = item["simulate_data"]["settings"]["region"]
            alpha_id = item.get("alpha_id")
            
            sub_universe_sharpe = 0.0
            robust_sharpe = 0.0
            ladder_yr_2_sharpe = 0.0
            returns = 0.0
            sharpe = 0.0
            fitness = 0.0
            turnover = 0.0
            margin = 0.0
            prod_correlation_val = 0.0

            if alpha_id:
                try:
                    result_json = self.safe_query_api(ace_lib.get_simulation_result_json, self.session, alpha_id)
                    if result_json:
                        is_data = result_json.get("is", {})
                        sharpe = float(is_data.get("sharpe", 0.0))
                        fitness = float(is_data.get("fitness", 0.0))
                        turnover = float(is_data.get("turnover", 0.0))
                        returns = float(is_data.get("returns", 0.0))
                        margin = float(is_data.get("margin", 0.0))
                        
                        sub_universe_sharpe = float(is_data.get("subUniverseSharpe", 0.0))
                        robust_sharpe = float(is_data.get("robustSharpe", 0.0))
                        
                        ladder_data = result_json.get("ladder", [])
                        if isinstance(ladder_data, list) and len(ladder_data) >= 2:
                            ladder_yr_2_sharpe = float(ladder_data[1].get("sharpe", 0.0))

                    # Query the production correlation directly from the server
                    prod_df = self.safe_query_api(ace_lib.get_prod_corr, self.session, alpha_id)
                    if prod_df is not None and not prod_df.empty and "max" in prod_df.columns:
                        live_corr = prod_df[prod_df.alphas > 0]
                        if not live_corr.empty:
                            prod_correlation_val = float(live_corr["max"].max())
                except Exception as e:
                    print(f"--> [Warning] Failed to process stats for {alpha_id}: {e}")
                    is_stats = item.get("is_stats")
                    if is_stats is not None and not is_stats.empty:
                        sharpe = float(is_stats.iloc[0].get("sharpe", 0.0))
                        fitness = float(is_stats.iloc[0].get("fitness", 0.0))
                        turnover = float(is_stats.iloc[0].get("turnover", 0.0))
                        margin = float(is_stats.iloc[0].get("margin", 0.0))

            is_tests = item.get("is_tests")
            checks_data = self.parse_is_checks(is_tests)
            
            passed_all_platform = self.check_if_passed(checks_data)
            
            # Local Genius-level validation check with strict custom thresholds
            passed_genius = (
                sharpe >= 1.58 and
                fitness >= 1.0 and
                0.01 <= turnover <= 0.40 and
                sub_universe_sharpe >= 1.0 and      # Enforcing sub universe is >= 1.0
                robust_sharpe >= 1.0 and
                ladder_yr_2_sharpe >= 2.02 and
                returns >= 0.08 and                 # Enforcing returns >= 8%
                prod_correlation_val < 0.60         # Enforcing prod correlation < 0.6
            )
            
            passed_count = self.get_passed_count(checks_data)
            corridor_bonus = 1 if (0.125 <= turnover <= 0.30) else 0

            record = {
                "generation": generation,
                "formula": formula,
                "region": region,
                "alpha_id": alpha_id,
                "sharpe": sharpe,
                "fitness": fitness,
                "turnover": turnover,
                "returns": returns,
                "prod_correlation": prod_correlation_val,
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
                    "region": region,
                    "alpha_id": alpha_id,
                    "sharpe": sharpe,
                    "fitness": fitness,
                    "turnover": turnover,
                    "returns": returns,
                    "prod_correlation": prod_correlation_val,
                    "margin_or_sub_sharpe": margin,
                    "sub_universe_sharpe": sub_universe_sharpe,
                    "robust_sharpe": robust_sharpe,
                    "ladder_year_2_sharpe": ladder_yr_2_sharpe,
                    "is_genius": "PASS" if passed_genius else "FAIL"
                }
                passed_record.update(checks_data)
                
                if not any(x["formula"] == formula and x["region"] == region for x in self.passed_alphas):
                    self.passed_alphas.append(passed_record)
                    status_lbl = "GENIUS" if passed_genius else "PLATFORM"
                    print(f"--> [{status_lbl} PASSED] [{region}] Alpha {alpha_id} passed! Added to golden list.")

            yield record

    def run_online_evolution(self, regional_seeds, num_generations=500):
        print(f"\n==============================================")
        print(f"STARTING GENIUS MULTI-REGION ISLAND MINER")
        print(f"==============================================\n")

        self.session = start_session()
        print("Session established successfully.")

        print("Querying allowed operators from WorldQuant Brain...")
        ops_df = get_operators(self.session)
        allowed_ops = set(ops_df['name'].tolist())

        # Load dynamic datafields for all islands/regions
        regional_fields = {}
        for r in self.regions:
            print(f"Querying allowed datafields for {r} region...")
            fields_df = get_datafields(self.session, region=r, delay=1, universe=self.universe_mapping[r])
            regional_fields[r] = fields_df

        self.mutator = CustomAlphaMutator(allowed_ops, regional_fields)
        print(f"Initialized Mutator for regions: {self.regions}")

        # Load seeds from passed list if it exists
        passed_path = "passed_alphas.csv"
        pre_existing_records = []
        if os.path.exists(passed_path):
            try:
                passed_df = pd.read_csv(passed_path)
                if not passed_df.empty and "formula" in passed_df.columns:
                    for _, row in passed_df.iterrows():
                        rec = row.to_dict()
                        rec["generation"] = 0
                        rec["status"] = "PASS"
                        rec["passed_count"] = int(rec.get("passed_count", 7))
                        rec["sub_universe_sharpe"] = float(rec.get("sub_universe_sharpe", 0.0))
                        rec["robust_sharpe"] = float(rec.get("robust_sharpe", 0.0))
                        rec["ladder_year_2_sharpe"] = float(rec.get("ladder_year_2_sharpe", 0.0))
                        rec["returns"] = float(rec.get("returns", 0.0))
                        rec["prod_correlation"] = float(rec.get("prod_correlation", 0.0))
                        rec["is_genius"] = rec.get("is_genius", "FAIL")
                        pre_existing_records.append(rec)
                    print(f"Loaded {len(pre_existing_records)} pre-existing alphas directly into memory.")
            except Exception as e:
                print(f"Note: Could not load past seeds: {e}")

        # Load the set of historically simulated formulas
        self.simulated_formulas = set()
        history_path = "gp_live_search_results.csv"
        if os.path.exists(history_path):
            try:
                hist_df = pd.read_csv(history_path)
                if not hist_df.empty and "formula" in hist_df.columns:
                    # Stored key format: (region, formula)
                    for _, row in hist_df.iterrows():
                        reg_val = str(row.get("region", "USA")).upper()
                        self.simulated_formulas.add((reg_val, row["formula"]))
                    print(f"Loaded {len(self.simulated_formulas)} historically simulated formulas.")
            except Exception as e:
                print(f"Note: Could not load history: {e}")

        # Distribute pre-existing records and clean seeds into separate region populations (islands)
        self.populations = {r: [] for r in self.regions}
        self.history.extend(pre_existing_records)
        self.passed_alphas = [r for r in pre_existing_records]

        for r in self.regions:
            # Add pre-existing passed records belonging to this region
            region_existing = [rec for rec in pre_existing_records if rec.get("region", "").upper() == r]
            self.populations[r].extend(region_existing)

            # Determine seeds for this region
            seeds = regional_seeds.get(r, [])
            cleaned_seeds = []
            for formula in seeds:
                cleaned = formula
                if "ts_delay" in allowed_ops:
                    cleaned = re.sub(r"\bdelay\(", "ts_delay(", cleaned)
                # Skip if already simulated
                if (r, cleaned) not in self.simulated_formulas and cleaned not in [x["formula"] for x in region_existing]:
                    cleaned_seeds.append(cleaned)

            if cleaned_seeds:
                print(f"Backtesting {len(cleaned_seeds)} new seed formulas for {r}...")
                seed_payloads = [self.build_payload(f, r) for f in cleaned_seeds]
                seed_results = self.safe_simulate_alpha_list(seed_payloads)
                new_seed_pop = list(self.process_simulation_results(seed_results, 0, "seed"))
                self.populations[r].extend(new_seed_pop)
                for f in cleaned_seeds:
                    self.simulated_formulas.add((r, f))

            # Crop initial population to target size
            self.populations[r] = sorted(self.populations[r], key=lambda x: (x.get("passed_count", 0), x.get("fitness", 0)), reverse=True)
            self.populations[r] = self.populations[r][:self.population_size_per_region]

        self.save_results_to_disk()

        # Evolution Loop
        for gen in range(1, num_generations + 1):
            self.keep_alive_session()
            print(f"\n======================================")
            print(f"EVOLVING GENERATION {gen}...")
            print(f"======================================")

            offspring_payloads = []
            generation_formulas = []  # List of tuples: (region, formula)

            # Evolve each region (island) independently
            for r in self.regions:
                region_pop = self.populations[r]
                if len(region_pop) < 2:
                    # Fallback to random templates if population is too small
                    print(f"--> [Warning] Pool for region {r} is too small to evolve. Seeding raw templates.")
                    raw_template_seed = [
                        "ts_mean(rank(close) * rank(volume), 252)",
                        "ts_mean(sign(open / close - 1) * scale(volume), 126)"
                    ]
                    for f in raw_template_seed:
                        offspring_payloads.append(self.build_payload(f, r))
                        generation_formulas.append((r, f))
                    continue

                region_offspring = []
                attempts = 0
                target_offspring_count = max(2, self.population_size_per_region // 2)

                while len(region_offspring) < target_offspring_count and attempts < 100:
                    attempts += 1
                    if random.random() < 0.7:
                        # Mutate
                        parent_record = self.select_parent_tournament(region_pop)
                        parent_formula = parent_record["formula"]
                        child = self.mutator.mutate(parent_formula, r, parent_record)
                        
                        if (child not in [p["formula"] for p in region_pop] and 
                            child not in region_offspring and 
                            (r, child) not in self.simulated_formulas):
                            region_offspring.append(child)
                    else:
                        # Crossover
                        parent_record1 = self.select_parent_tournament(region_pop)
                        parent_record2 = self.select_parent_tournament(region_pop)
                        parent1 = parent_record1["formula"]
                        parent2 = parent_record2["formula"]
                        
                        if parent1 != parent2:
                            child1, child2 = CustomAlphaMutator.crossover(parent1, parent2)
                            for child in (child1, child2):
                                if (len(region_offspring) < target_offspring_count and 
                                    child not in [p["formula"] for p in region_pop] and 
                                    child not in region_offspring and 
                                    (r, child) not in self.simulated_formulas):
                                    region_offspring.append(child)

                # Add to total batch
                for f in region_offspring:
                    offspring_payloads.append(self.build_payload(f, r))
                    generation_formulas.append((r, f))

            if not offspring_payloads:
                print(f"--> [Warning] Evolved pool in generation {gen} was already completely simulated. Stopping evolution.")
                break

            print(f"Submitting {len(offspring_payloads)} offspring formulas across all regions to simulator...")
            offspring_results = self.safe_simulate_alpha_list(offspring_payloads)
            new_candidates = list(self.process_simulation_results(offspring_results, gen, "evolved"))

            # Update simulated history in memory
            for r, f in generation_formulas:
                self.simulated_formulas.add((r, f))

            # Apply selection on each island separately
            for r in self.regions:
                region_new = [c for c in new_candidates if c["region"] == r]
                combined_pool = self.populations[r] + region_new
                combined_pool = sorted(combined_pool, key=lambda x: (x.get("passed_count", 0), x.get("fitness", 0)), reverse=True)
                self.populations[r] = combined_pool[:self.population_size_per_region]

            self.save_results_to_disk()
            print(f"Generation {gen} completed. Passed alphas count: {len(self.passed_alphas)}")

    def save_results_to_disk(self):
        history_df = pd.DataFrame(self.history)
        history_df.to_csv("gp_live_search_results.csv", index=False)
        
        passed_df = pd.DataFrame(self.passed_alphas)
        if passed_df.empty:
            passed_df = pd.DataFrame(columns=["formula", "region", "alpha_id", "sharpe", "fitness", "turnover", "margin_or_sub_sharpe", "sub_universe_sharpe", "robust_sharpe", "ladder_year_2_sharpe", "is_genius", "sharpe_test", "fitness_test", "turnover_min_test", "turnover_max_test", "weight_test", "sub_sharpe_test", "challenge_test"])
        passed_df.to_csv("passed_alphas.csv", index=False)
        
        # Copy to Downloads
        import shutil
        try:
            shutil.copy2("gp_live_search_results.csv", r"C:\Users\HP\Downloads\gp_live_search_results_nyumuharon.csv")
            shutil.copy2("passed_alphas.csv", r"C:\Users\HP\Downloads\passed_alphas_nyumuharon.csv")
            shutil.copy2("gp_genius_miner.log", r"C:\Users\HP\Downloads\gp_genius_miner_nyumuharon.log")
        except Exception:
            pass

def input_with_timeout(prompt, timeout=10, default="ASI"):
    import sys
    import time
    try:
        import msvcrt
        sys.stdout.write(prompt)
        sys.stdout.flush()
        
        start_time = time.time()
        input_str = ""
        
        while True:
            if time.time() - start_time > timeout:
                sys.stdout.write("\n")
                sys.stdout.flush()
                print(f"--> Timeout reached. Using default: {default}")
                return default
                
            if msvcrt.kbhit():
                try:
                    char = msvcrt.getwche()
                    if char in ('\r', '\n'):
                        sys.stdout.write("\n")
                        sys.stdout.flush()
                        return input_str.strip()
                    elif char == '\b':
                        if len(input_str) > 0:
                            input_str = input_str[:-1]
                            sys.stdout.write(" \b")
                            sys.stdout.flush()
                    else:
                        input_str += char
                except Exception:
                    pass
            time.sleep(0.05)
    except Exception:
        try:
            val = input(prompt).strip()
            return val if val else default
        except Exception:
            return default

if __name__ == "__main__":
    region_choice = input_with_timeout(
        "Enter target region to mine (ASI, USA, GLB, IND, or ALL) [default: ASI]: ",
        timeout=10,
        default="ASI"
    )
    if region_choice not in ("ASI", "USA", "GLB", "IND", "ALL"):
        print(f"Unknown region '{region_choice}'. Defaulting to ASI.")
        region_choice = "ASI"

    # Load dynamic regional seed pools based on your passed alphas
    regional_seeds = {
        "USA": [
            "rank(anl4_capex_flag)-rank(assets/liabilities_curr) + (0.59*trade_when(option_breakeven_270>put_breakeven_270,-rank(fn_eff_income_tax_rate_continuing_operations_a),-1)) + -ts_corr(ts_delta(fscore_momentum,90),ts_delta(fscore_value,90),920)*rank(-volume)",
            "ts_mean(rank(close) * rank(volume), 252)",
            "ts_mean(sign(open / close - 1) * scale(volume), 126)"
        ],
        "GLB": [
            "ts_mean(rank(close) * rank(volume), 252)",
            "ts_mean(sign(open / close - 1) * scale(volume), 126)",
            "ts_delay(rank(close) * ts_rank(volume, 126), 63)"
        ],
        "IND": [
            "alpha=ts_scale(oth335_hc_combined_all_region_linear,400)+ts_scale(mdl110_score,252);group_neutralize(rank(ts_backfill(alpha,600))+ts_rank(-returns,252),subindustry);"
        ],
        "ASI": [
            "alpha=ts_rank(imb5_score,400)+ts_rank(rel_val_buyback_yield_component_score_3,252);group_neutralize(rank(ts_backfill(alpha,600))+ts_rank(-returns,252),subindustry);",
            "alpha=ts_rank(mdl110_score,400)+ts_rank(rel_val_buyback_yield_component_score_3*adv20,252);group_neutralize(rank(ts_backfill(alpha,600))+ts_rank(-returns,252),subindustry);"
        ]
    }
    
    if region_choice == "ALL":
        target_regions = ["USA", "GLB", "IND", "ASI"]
        concurrency = 4
        pop_size = 6
    else:
        target_regions = [region_choice]
        concurrency = 4  # Limit to 4 concurrent simulations as requested
        pop_size = 12    # Increase population size to 12 for deep single-region evolution
        
    print(f"--> Starting Evolution Loop for regions: {target_regions} (Concurrency: {concurrency}, Pop Size: {pop_size})")
    gp = WQOnlineGP(regions=target_regions, population_size_per_region=pop_size)
    gp.concurrency_limit = concurrency
    gp.run_online_evolution(regional_seeds, num_generations=500)
