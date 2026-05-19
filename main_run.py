import json
import random
import time
import os
import logging
import math
from datetime import datetime
from prompt_universal import generate_agent_prompt  # Ensure your prompt file is named this
from LLM_call import LLMInterface

# Top of main_run.py

DAILY_CORPUS = {
    "Student": ["So tired of the 8 AM class, want to sleep.", "Too much homework, going bald.",
                "The new dish in the cafeteria is pretty good.",
                "Finally class is over, going back to the dorm to play games!"],
    "Teacher": ["Just finished grading homework, ready to get off work.",
                "Have to prepare for a public lecture next week, mountain of pressure.",
                "Three classes in a row, my throat is smoking."],
    "Farmer": ["Looking at the weather it should rain tomorrow, good thing I don't need to water the fields.",
               "The harvest this year should be okay.", "Working on the farm all day, back aches."],
    "Civil Servant": ["Another day of writing materials, headache.",
                      "Just finished a meeting, a bunch of directives need to be implemented.",
                      "Off work on time, life is good."],
    "Factory Worker": ["The assembly line is moving so fast today.",
                       "Finally made it to the shift change, going to the cafeteria.",
                       "Got paid, treating myself tonight!"],
    "Doctor": ["Saw so many patients in the clinic today, no time to drink water.",
               "Doctor-patient communication is truly an art.", "Night shift, pray for peace."],
    "Business Owner": ["Business is hard, but still have to persist.",
                       "It's the end of the month, looking at the financial statements makes me worry.",
                       "Closed a small deal today."],
    "Freelancer": ["The client asked to change the requirements again, it's already the fifth version!",
                   "Freelancing means being your own boss.", "Finally submitted the draft, lying flat for two days."],
    "Retiree": ["Did some Tai Chi in the park, feeling refreshed.", "Young people nowadays work too hard.",
                "Going to pick up my grandson from school in the afternoon."],
    "Software Engineer": ["Code runs! Zero bugs!", "Another day of fixing legacy code.",
                          "Planning not to work overtime today, time to run!"]
}


# ==========================================
# 1. Logging Configuration
# ==========================================
def setup_logging(results_dir):
    logger = logging.getLogger("SocialSimulation")
    logger.setLevel(logging.DEBUG)

    # ==========================================
    # [New]: Clear old log handlers to prevent duplicate logging during Monte Carlo loops
    # ==========================================
    if logger.hasHandlers():
        logger.handlers.clear()

    log_path = os.path.join(results_dir, "simulation.log")
    file_handler = logging.FileHandler(log_path, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler()
    # [Modified]: Change INFO to DEBUG to ensure the console outputs detailed node tracking info
    console_handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


# ==========================================
# 2. Simulation Core Engine
# ==========================================
class SocialSimulation:
    def __init__(self, config):
        self.config = config
        self.current_round = 0
        self.simulation_start_time = datetime.now()
        self.timestamp = self.simulation_start_time.strftime("%Y%m%d-%H%M%S")
        self.results_dir = f"simulation_results_{self.timestamp}"
        os.makedirs(self.results_dir, exist_ok=True)

        self.logger = setup_logging(self.results_dir)
        self.logger.info(f"=== Complex Network Multi-Agent Social Simulation System Initialization ===")

        # ==========================================
        # [New]: Set global random seed to ensure complete reproducibility
        # ==========================================
        self.seed = config.get('seed', 46)  # Default to 46 if not provided
        random.seed(self.seed)
        self.logger.info(f"Global random seed set to: {self.seed}")

        # Load data and interfaces
        self.agents = self.load_agents(config['agent_file'])
        self.total_agents = len(self.agents)
        self.llm_interface = LLMInterface(
            model_provider=config['model_provider'],
            api_key=config['api_key'],
            model_name=config['model_name'],
            temperature=config.get('temperature', 0.7)
        )

        # State containers
        self.current_posts = {}
        self.agent_beliefs = {aid: False for aid in self.agents}
        self.post_history = []

        self.simulation_results = {
            "config": config,
            "start_time": self.simulation_start_time.isoformat(),
            "rounds": []
        }

        self.initialize_posts()

    def load_agents(self, file_path):
        self.logger.info(f"Loading network agent data: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data['agents']

    def initialize_posts(self):
        """Initialize world and seed node injection"""
        self.logger.info("Initializing timeline and rumor seed nodes...")
        neutral_posts = ["The weather is really nice today.", "Work is so tiring, want a vacation.",
                         "Planning to watch a movie this weekend."]
        self.initial_rumor = self.config.get('initial_rumor',
                                             "Urgent notice! Internal news says there will be massive layoffs soon, run!")

        # 1. Round 0: Neutral posts from everyone
        for agent_id in self.agents:
            initial_post = random.choice(neutral_posts)
            self.current_posts[agent_id] = initial_post
            self.post_history.append((0, agent_id, initial_post))

        # 2. Strategically select seed nodes
        all_agents = list(self.agents.keys())
        num_seeds = max(1, int(self.total_agents * self.config.get('initial_spreader_ratio', 0.05)))
        strategy = self.config.get('seed_strategy', 'random')

        if strategy == 'structural':
            self.logger.info("Using [Topological Centrality Influencer] strategy to select seeds...")
            sorted_by_degree = sorted(all_agents, key=lambda aid: len(self.agents[aid].get("neighbors", [])),
                                      reverse=True)
            candidate_spreaders = sorted_by_degree[:num_seeds]
        elif strategy == 'psychological':
            self.logger.info("Using [Psychologically Susceptible] strategy to select seeds...")

            def psy_score(aid):
                psy = self.agents[aid]["psychology"]["big_five"]
                return psy["Extraversion"] + psy["Neuroticism"]

            sorted_by_psy = sorted(all_agents, key=psy_score, reverse=True)
            candidate_spreaders = sorted_by_psy[:num_seeds]
        else:
            self.logger.info("Using [Random Sampling] strategy to select seeds...")
            candidate_spreaders = random.sample(all_agents, num_seeds)

        # 3. Infect seeds
        for agent_id in candidate_spreaders:
            self.current_posts[agent_id] = self.initial_rumor
            self.agent_beliefs[agent_id] = True
            self.post_history = [(r, aid, p) if aid != agent_id else (0, aid, self.initial_rumor) for r, aid, p in
                                 self.post_history]

    def run_simulation(self, rounds=3):
        self.logger.info(f"\n========== Starting Multi-Agent Simulation Evolution ==========")
        for round_idx in range(rounds):
            self.current_round = round_idx + 1
            start_time = time.time()
            self.logger.info(f"\n► Executing round: {self.current_round} / {rounds}")

            round_result = self.run_round()
            self.simulation_results["rounds"].append(round_result)
            self.save_round_snapshot(round_result)

            duration = time.time() - start_time
            stats = round_result["summary"]

            # ==========================================
            # [Fixed here]: Align key names with the latest 4-quadrant statistical metrics
            # ==========================================
            self.logger.info(
                f"[Round {self.current_round} Summary] Time: {duration:.1f}s | "
                f"Penetration (Believe): {stats['penetration_rate']:.1%} | "
                f"Skepticism (Disbelieve): {stats['skeptics_ratio']:.1%} | "
                f"Sys Entropy: {stats['system_entropy']:.3f} | "
                f"Bipolarization: {stats['bipolarization_index']:.3f}"
            )

        self.save_final_results()
        self.logger.info("\n========== Social Simulation Successfully Completed ==========")
        return self.simulation_results

    def run_round(self):
        """Synchronous state update with memory"""
        agent_ids = list(self.agents.keys())
        random.shuffle(agent_ids)

        round_data = {
            "round": self.current_round,
            "agent_decisions": {},
            "summary": {}
        }

        # [Important] Generate static world snapshot
        snapshot_posts = self.current_posts.copy()
        next_round_posts = {}

        # State statistics counter
        counts = {"active_spreaders": 0, "silent_believers": 0, "debunkers": 0, "immune_neutrals": 0}

        for idx, agent_id in enumerate(agent_ids):
            agent = self.agents[agent_id]
            agent_history = [(r, p) for r, aid, p in self.post_history if aid == agent_id]

            # Use eye-catching separators to distinguish different node processing
            self.logger.debug(f"\n{'-' * 80}")
            self.logger.info(
                f"▶ [Round {self.current_round}] Processing node {idx + 1}/{self.total_agents} (ID: {agent_id} - {agent['demographics']['name']})")
            '''
            try:
                # 1. Assemble Prompt
                prompt_data = generate_agent_prompt(agent_id, self.agents, snapshot_posts, agent_history)

                # [New Tracking Log: State Printing]
                self.logger.debug(f"  [State 1] Own post history: {prompt_data['self_history']}")
                self.logger.debug(f"  [State 2] Seen neighbor posts: {prompt_data.get('seen_posts', [])}")
                self.logger.debug(f"  [State 3] Sent Prompt:\n{prompt_data['prompt']}")

                # 2. Request LLM (This will trigger raw return printing in LLM_call.py)
                response = self.llm_interface.call_llm(prompt_data["prompt"])

                # [New Tracking Log: Parsed Result Printing]
                self.logger.debug(f"  [State 4] Final parsed Agent behavior dictionary: {json.dumps(response, ensure_ascii=False)}")

                is_believed = response.get("is_believed", False)
                will_spread = response.get("will_spread", False)
                new_post = response.get("new_post", "")

                # 3. State Record
                self.agent_beliefs[agent_id] = is_believed
                if will_spread and new_post:
                    next_round_posts[agent_id] = new_post
                    self.post_history.append((self.current_round, agent_id, new_post))
                    self.logger.info(f"  ★ [{agent['demographics']['name']}] decided to post: {new_post}")
                else:
                    self.logger.debug(f"  ☆ [{agent['demographics']['name']}] stays silent.")

                # 3. Faction Classification
                if is_believed and will_spread:
                    counts["active_spreaders"] += 1
                elif is_believed and not will_spread:
                    counts["silent_believers"] += 1
                elif not is_believed and will_spread:
                    counts["debunkers"] += 1
                else:
                    counts["immune_neutrals"] += 1

                round_data["agent_decisions"][agent_id] = {"response": response}

            except Exception as e:
                self.logger.error(f"Error processing agent {agent_id}: {str(e)}") 
            '''

            try:
                # 1. Get model response
                prompt_data = generate_agent_prompt(agent_id, self.agents, snapshot_posts, agent_history)

                # [Critical Fix]: Unconditionally get the raw prompt first to ensure the variable exists!
                final_prompt = prompt_data["prompt"]

                # ==========================================
                # [Minimal Modification]: Forcibly insert debunking info into the prompt only in specific rounds
                # ==========================================
                if self.config.get("enable_debunking", False):
                    official_message = ""
                    # According to your plan, deploy in rounds 6-7, 9-10, 12-13
                    if 7 <= self.current_round <= 10:
                        official_message = (
                            "Water Quality Monitoring Center: The east city water source is NOT polluted, all water quality tests are normal!\n"
                            "[Government News] 72-hour continuous monitoring has been conducted at 15 sampling points across the city, data shows water quality is safe.\n"
                            "[Water Quality Monitoring Center] All 106 water quality indicators passed (Test No: 2026WS001). Do not believe rumors.\n")
                    elif 11 <= self.current_round <= 15:
                        official_message = (
                            "Water Quality Monitoring Center: The east city water source is NOT polluted, all water quality tests are normal!\n"
                            "[Government News] 72-hour continuous monitoring has been conducted at 15 sampling points across the city, data shows water quality is safe.\n"
                            "[Water Quality Monitoring Center] All 106 water quality indicators passed (Test No: 2026WS001). Do not believe rumors.\n"
                            )

                    if official_message:
                        inject_text = f"\n\n### Social Media:\n{official_message}\n"
                        target_str = "### Currently, the following latest updates appeared on your social network:"
                        # Cleverly insert debunking info before social circle updates
                        if target_str in final_prompt:
                            final_prompt = final_prompt.replace(target_str, inject_text + target_str)
                        else:
                            final_prompt = inject_text + final_prompt

                # ==========================================
                # [New]: States 1~3, detailed logging of prior info input to LLM
                # ==========================================
                self.logger.debug(f"  [State 1] Own post history: {prompt_data.get('self_history', [])}")
                self.logger.debug(f"  [State 2] Seen neighbor posts: {prompt_data.get('seen_posts', [])}")
                self.logger.debug(f"  [State 3] Complete Sent Prompt:\n{final_prompt}")

                response = self.llm_interface.call_llm(final_prompt)

                # ==========================================
                # [New]: State 4, detailed logging of the raw parsed result returned by the LLM
                # ==========================================
                self.logger.debug(
                    f"  [State 4] Final parsed behavior dictionary: {json.dumps(response, ensure_ascii=False)}")

                is_believed = response.get("is_believed", False)
                will_spread = response.get("will_spread", False)
                new_post = str(response.get("new_post", "")).strip()
                occupation = agent['demographics']['occupation']

                # --- Logic Patch: Post-filtering and local corpus intervention ---

                # Core event keywords (dynamically adjusted based on your initial_rumor)
                core_keywords = ["water", "polluted", "urgent", "notice", "pollution"]

                # Check: If Spread but content lacks keywords, deem it a "false positive", force to daily routine
                if will_spread and not any(kw in new_post.lower() for kw in core_keywords):
                    will_spread = False
                    is_believed = False  # Core event not mentioned, deemed not brainwashed

                # If not spreading (or an intercepted false positive), fetch daily routine from local corpus
                if not will_spread:
                    # Get corpus for occupation, use general corpus if none available
                    corpus = DAILY_CORPUS.get(occupation, ["Another ordinary day today.", "Working hard..."])
                    new_post = random.choice(corpus)

                # --- State Record ---
                self.agent_beliefs[agent_id] = is_believed

                # Whether spreading or daily, post as long as there is content
                if new_post:
                    next_round_posts[agent_id] = new_post
                    self.post_history.append((self.current_round, agent_id, new_post))

                    if will_spread:
                        self.logger.info(f"  ★ [{agent['demographics']['name']}] Spreading core message: {new_post}")
                    else:
                        self.logger.info(f"  ☕ [{agent['demographics']['name']}] Posting daily routine: {new_post}")

                # ==========================================
                # [Modification 1]: Brand new four-quadrant faction statistics
                # ==========================================
                if is_believed and will_spread:
                    counts["active_spreaders"] += 1  # Active spreaders
                elif is_believed and not will_spread:
                    counts["silent_believers"] += 1  # Silent believers
                elif not is_believed and will_spread:
                    counts["debunkers"] += 1  # Active debunkers (very few)
                else:
                    counts["immune_neutrals"] += 1  # Rational skeptics/ignoring (vast majority)

            except Exception as e:
                self.logger.error(f"Error processing agent {agent_id}: {str(e)}")

        # [Important] All agent decisions completed, uniformly refresh public screen
        self.current_posts.update(next_round_posts)

        # ==========================================
        # 4. Calculate core metrics of communication and social physics (4-quadrant version)
        # ==========================================
        total = self.total_agents
        # Calculate proportions of the 4 quadrants respectively
        p_active_spread = counts["active_spreaders"] / total
        p_silent_belief = counts["silent_believers"] / total
        p_active_debunk = counts["debunkers"] / total
        p_passive_skeptic = counts["immune_neutrals"] / total

        # --- Metric 1: 4-State System Shannon Entropy ---
        # Measures opinion and behavior fragmentation of society (Max 2.0)
        entropy = 0
        for p in [p_active_spread, p_silent_belief, p_active_debunk, p_passive_skeptic]:
            if p > 0:
                entropy -= p * math.log2(p)

        # --- Metric 2: Bipolarization Index ---
        # Measures antagonism between 'Believer Faction (inc. silent)' and 'Skeptic Faction (inc. silent)'
        total_believers_ratio = p_active_spread + p_silent_belief
        total_skeptics_ratio = p_active_debunk + p_passive_skeptic
        # Closer to 1 means evenly matched, closer to 0 means one-sided
        bipolarization = 1.0 - abs(total_believers_ratio - total_skeptics_ratio)

        # --- Metric 3: Penetration Rate ---
        penetration_rate = total_believers_ratio

        round_data["summary"] = {
            # Absolute quantity details
            "active_spreaders": counts["active_spreaders"],
            "silent_believers": counts["silent_believers"],
            "active_debunkers": counts["debunkers"],
            "passive_skeptics": counts["immune_neutrals"],  # Previously neutrals, now silent skeptics

            # Proportion metrics
            "believers_ratio": total_believers_ratio,
            "skeptics_ratio": total_skeptics_ratio,

            # Academic-level core metrics
            "system_entropy": entropy,  # System chaos (0~2)
            "bipolarization_index": bipolarization,  # Faction bipolarization degree (0~1)
            "penetration_rate": penetration_rate  # Rumor penetration rate
        }

        # Console concise print
        self.logger.info(
            f"[Round {self.current_round}] Penetration: {penetration_rate:.1%} | "
            f"Skepticism: {total_skeptics_ratio:.1%} | "
            f"Entropy: {entropy:.3f} | Bipolarity: {bipolarization:.3f}"
        )

        return round_data

    def save_round_snapshot(self, round_data):
        path = os.path.join(self.results_dir, f"round_{self.current_round}_{self.timestamp}.json")
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(round_data, f, ensure_ascii=False, indent=2)

    def save_final_results(self):
        with open(os.path.join(self.results_dir, f"simulation_summary_{self.timestamp}.json"), 'w',
                  encoding='utf-8') as f:
            json.dump(self.simulation_results, f, ensure_ascii=False, indent=2)
        with open(os.path.join(self.results_dir, f"post_history_{self.timestamp}.json"), 'w', encoding='utf-8') as f:
            json.dump(self.post_history, f, ensure_ascii=False, indent=2)


# ==========================================
# 3. Execution Entry
# ==========================================
if __name__ == "__main__":
    config = {
        "agent_file": "net_ba_100_seed46.json",
        "model_provider": "qwen",
        "api_key": "sk-ea17a3710a9f4ebaa5bfb4d1ac4e87fd",
        "model_name": "qwen-turbo",
        "temperature": 0.7,
        "rounds": 15,
        "seed_strategy": "structural",  # Optional: 'structural', 'psychological', 'random'
        "initial_spreader_ratio": 0.05,
        "initial_rumor": "Urgent Notice: The east city water source is polluted, do not drink tap water!",
        "enable_debunking": True,
    }

    simulation = SocialSimulation(config)
    simulation.run_simulation(rounds=config["rounds"])