import os
import re
import random
import pandas as pd
from alpha_generator import REGIONS, BOOSTER_OPERATORS, SUPPORTING_OPERATORS, LOOKBACKS, apply_operator

class AlphaMutator:
    def __init__(self, region="USA"):
        self.region = region
        # Fallback to GLB if region is not specifically configured
        self.region_config = REGIONS.get(region, REGIONS["GLB"])
        
        # Flatten all fields for easy lookup and mutation
        self.all_fields = []
        self.field_to_category = {}
        for cat, fields in self.region_config["supporting"].items():
            for f in fields:
                self.all_fields.append(f)
                self.field_to_category[f] = cat
                
        self.booster_fields = self.region_config["booster"]

    def mutate_fields(self, formula):
        """Finds any field in the formula and replaces it with another from the same category or booster list."""
        mutated = formula
        
        # 1. Mutate supporting fields
        for field in self.all_fields:
            if field in formula:
                cat = self.field_to_category[field]
                choices = [f for f in self.region_config["supporting"][cat] if f != field]
                if choices:
                    new_field = random.choice(choices)
                    # Use regex with word boundaries to avoid partial replacement (e.g. md138 replacing md1382)
                    mutated = re.sub(rf"\b{field}\b", new_field, mutated)
                    
        # 2. Mutate booster fields
        for field in self.booster_fields:
            if field in formula:
                choices = [f for f in self.booster_fields if f != field]
                if choices:
                    new_field = random.choice(choices)
                    mutated = re.sub(rf"\b{field}\b", new_field, mutated)
                    
        return mutated

    def mutate_operators(self, formula):
        """Finds operators in the formula and replaces them with another of the same class."""
        mutated = formula
        
        # Mutate booster operators
        for op_name, _ in BOOSTER_OPERATORS:
            if op_name + "(" in formula:
                choices = [op[0] for op in BOOSTER_OPERATORS if op[0] != op_name]
                new_op = random.choice(choices)
                mutated = re.sub(rf"\b{op_name}\(", f"{new_op}(", mutated)
                
        # Mutate supporting operators
        for op_name, _ in SUPPORTING_OPERATORS:
            if op_name + "(" in formula:
                choices = [op[0] for op in SUPPORTING_OPERATORS if op[0] != op_name]
                new_op = random.choice(choices)
                mutated = re.sub(rf"\b{op_name}\(", f"{new_op}(", mutated)
                
        return mutated

    def mutate_lookbacks(self, formula):
        """Finds lookback periods and replaces them with another lookback from the configuration."""
        mutated = formula
        for val in LOOKBACKS:
            # Use regex to find lookback numbers as separate arguments (e.g., ", 126)")
            pattern = rf",\s*{val}\)"
            if re.search(pattern, formula):
                choices = [v for v in LOOKBACKS if v != val]
                new_val = random.choice(choices)
                mutated = re.sub(pattern, f", {new_val})", mutated)
        return mutated

    def mutate(self, formula):
        """Applies a random mutation type (Field, Operator, or Lookback)."""
        mutation_type = random.choice(["field", "operator", "lookback"])
        if mutation_type == "field":
            return self.mutate_fields(formula)
        elif mutation_type == "operator":
            return self.mutate_operators(formula)
        else:
            return self.mutate_lookbacks(formula)

    @staticmethod
    def crossover(formula1, formula2):
        """Crossover swaps the supporting part of two formulas split by the multiplication '*' operator."""
        if "*" not in formula1 or "*" not in formula2:
            return formula1 # Fallback
            
        parts1 = [p.strip() for p in formula1.split("*", 1)]
        parts2 = [p.strip() for p in formula2.split("*", 1)]
        
        # Swap components: Part1 of F1 * Part2 of F2
        child1 = f"{parts1[0]} * {parts2[1]}"
        child2 = f"{parts2[0]} * {parts1[1]}"
        
        return child1, child2


class GPSearchEngine:
    def __init__(self, region="USA"):
        self.region = region
        self.mutator = AlphaMutator(region=region)

    def run_search_offline(self, seeds, generations=3, population_size=10):
        print(f"\n--- Starting Offline GP Search for Region: {self.region} ---")
        print(f"Initial Seed Alphas: {len(seeds)}")
        
        current_population = list(seeds)
        all_results = []
        
        # Save initial seeds
        for idx, formula in enumerate(seeds):
            all_results.append({
                "generation": 0,
                "id": f"gen0_seed{idx}",
                "formula": formula,
                "origin": "seed",
                "sharpe": None,
                "fitness": None
            })
            
        for gen in range(1, generations + 1):
            print(f"Evolving Generation {gen}...")
            new_generation = []
            
            while len(new_generation) < population_size:
                # 80% chance of mutation, 20% chance of crossover
                if random.random() < 0.8:
                    parent = random.choice(current_population)
                    child = self.mutator.mutate(parent)
                    # Check for duplicates or invalid formulas
                    if child not in current_population and child not in new_generation:
                        new_generation.append(child)
                        all_results.append({
                            "generation": gen,
                            "id": f"gen{gen}_mut{len(new_generation)}",
                            "formula": child,
                            "origin": f"mutation_of_{parent}",
                            "sharpe": None,
                            "fitness": None
                        })
                else:
                    parent1 = random.choice(current_population)
                    parent2 = random.choice(current_population)
                    if parent1 != parent2:
                        child1, child2 = AlphaMutator.crossover(parent1, parent2)
                        for child in (child1, child2):
                            if len(new_generation) < population_size and child not in current_population and child not in new_generation:
                                new_generation.append(child)
                                all_results.append({
                                    "generation": gen,
                                    "id": f"gen{gen}_cross{len(new_generation)}",
                                    "formula": child,
                                    "origin": f"crossover_of_{parent1}_and_{parent2}",
                                    "sharpe": None,
                                    "fitness": None
                                })
            
            # Since we are offline, all offspring automatically survive to become the new generation seeds
            current_population = new_generation
            
        df = pd.DataFrame(all_results)
        return df

if __name__ == "__main__":
    # Our selected top 3 best formulas (using GLB region as our base config)
    seed_formulas = [
        "ts_mean(rank(snt21), 252) * rank(oth335)",
        "ts_mean(sigmoid(ern3), 126) * scale(md138)",
        "delay(rank(pvi), 63) * ts_rank(md138, 126)"
    ]
    
    # Run GP search offline for GLB region
    engine = GPSearchEngine(region="GLB")
    results_df = engine.run_search_offline(seed_formulas, generations=3, population_size=10)
    
    # Save results to workspace folder
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, "gp_search_results.csv")
    results_df.to_csv(output_path, index=False)
    
    print(f"\nGP Search complete. Total formulas in history: {len(results_df)}")
    print(f"Results saved to: {output_path}")
    
    # Show summary of results
    print("\nMutated & Evolved Formula Examples:")
    print(results_df[results_df["generation"] > 0][["generation", "id", "formula", "origin"]].head(10).to_string())
