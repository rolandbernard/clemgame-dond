A Negotiation Game: Deal or No Deal (DoND)
==========================================

This repository includes the implementation of a new [clemgame](https://clembench.github.io/) implementing a negotiation based dialogue game. This repository also contains the raw results of an evaluation performed using a number of recent closed- and open-weight models. The project investigates the negotiation capabilities of various large language models (LLMs) by employing a clemgame based on the Deal or No Deal (DoND) negotiation game. The core objective is to understand how well LLMs are able to cooperate and compromise in order to reach mutually beneficial agreements.

This clemgame challenges LLMs in several key areas:

* **Grounding**: Performance is negatively impacted if LLMs invent new items or misrepresent value functions, ensuring they stay grounded in the provided game state.
* **Communication**: LLMs must clearly articulate their preferences and understand those expressed by their opponent.
* **Agreement Generation**: The game requires LLMs to propose and agree upon mutually acceptable solutions.

## Setup and Installation

This project was developed and tested with **Python 3.10**. It is highly recommended to set up a virtual environment to manage dependencies.

1. **Clone the repository**:
    ```bash
    git clone https://github.com/rolandbernard/clemgame-dond
    cd clemgame-dond
    ```
2. **Create and activate a virtual environment**:
    The `run_benchmark.sh` script is configured to look for a virtual environment located at `./venv/`.
    ```bash
    python3.10 -m venv venv
    source venv/bin/activate
    ```
3. **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## API Key and Model Configuration

To interact with the LLMs, you will need to configure your API keys and model definitions. This project follows the standard `clembench` setup.

1. **API Keys (`key.json.example`)**:
    The `key.json.example` file provides an example of how to set up the API keys for various model providers used during the evaluation (Google, Mistral, Azure, and groq).
      * Copy this file and rename it to `key.json`:
        ```bash
        cp key.json.example key.json
        ```
      * Open `key.json` and replace the placeholder values with your actual API keys for the services you intend to use.
2. **Model Definitions (`model_registry.json`)**:
    The `model_registry.json` file contains the definitions of the specific LLMs used in the evaluation. These definitions reference the backends for which API keys are configured in `key.json`. You can modify this file to include other models or specific backends or versions you wish to test.

## Running the Evaluation

The `run_benchmark.sh` script is the primary entry point for executing the evaluation benchmarks performed for this project.

1. **Ensure virtual environment is present**:
    The virtual environment does not necessarily need to be active, as the script will automatically activate it in any case. However, you should make sure that you have created it and installed the dependencies as indicated above. The script will use `./venv/` by default.
2. **Execute the benchmark script**:
    ```bash
    bash run_benchmark.sh
    ```
    **Note**: The `run_benchmark.sh` script does not accept command-line parameters directly. However, you can modify the constants defined at the beginning of the script (e.g., `models`, `languages`, `modes`) to customize which models, languages, or game modes are included in your specific benchmark run.

## Project Structure

Here's an overview of the key files and directories in this project:

* `dond/`: The core `clemgame` implementation for Deal or No Deal.
    * `clemgame.json`: Defines the game in the standard `clembench` format.
    * `instancegenerator.py`: Script used to programmatically generate new game instances.
    * `master.py`: Contains the main logic for the game master, the game scorer, and the player agents within the game.
    * `in/`: Contains game instance files.
        * `instances_comp_de.json`, `instances_comp_en.json`, `instances_comp_it.json`: Competitive mode instances for German, English, Italian.
        * `instances_coop_de.json`, `instances_coop_en.json`, `instances_coop_it.json`: Cooperative mode instances for German, English, Italian.
        * `instances_semi_de.json`, `instances_semi_en.json`, `instances_semi_it.json`: Semi-competitive mode instances for German, English, Italian.
        * `instances.json`: A smaller set of two instances, primarily intended for testing purposes.
    * `resources/`: Contains various templates and data used by `instancegenerator.py`.
        * `de/`, `en/`, `it/`: Language-specific resources.
            * `initial_comp.template`, `initial_coop.template`, `initial_semi.template`: Initial prompts given to LLMs explaining the game rules for competitive, cooperative, and semi-competitive modes respectively.
            * `proposal_early.template`: Prompt used when the other player makes a proposal.
            * `proposal_timeout.template`: Prompt used when the turn limit is reached.
            * `possible_items.json`, `possible_items_plural.json`: Lists of different item types (singular and plural forms) from which new game instances are sampled.
* `azure_api.py`: Contains a `clembench` backend implementation to integrate with the [Azure OpenAI API](https://azure.microsoft.com/en-us/products/ai-services/openai-service), which is not supported by default in `clembench`.
* `key.json.example`: An example file demonstrating the expected format for API keys to access different LLM providers. Copy and rename to `key.json` and fill with your actual keys.
* `model_registry.json`: Defines the configurations of the LLMs used during the evaluation.
* `run_benchmark.sh`: The main shell script to execute all defined benchmarks for the project.
* `results/`: Directory storing the raw and processed results of the evaluation runs.
    * Subdirectories: Contains one subdirectory for each tested model, further nested by mode/language combinations, and then individual episode results.
    * `raw.csv`: A summary CSV file containing the scores and metrics achieved in each game episode.
* `LICENSE`: The license under which this project is distributed.

## Evaluation Metrics

The `results/raw.csv` file provides a table of the game outcomes and key metrics captured during the evaluation. The following are the main metrics for assessing various aspects of the LLMs' negotiation performance:

* **`Aborted`**: A boolean (0/1) flag indicating if the game was aborted due to LLMs not following rules.
* **`Lose`**: A boolean (0/1) flag indicating if no valid agreement was reached.
* **`Success`**: A boolean (0/1) flag indicating if a valid agreement was successfully reached.
* **`Pareto Optimal`**: A boolean (0/1) flag indicating whether the final agreement reached by the LLMs is Pareto optimal, i.e., no player's score can be improved without decreasing another player's score.
* **`Main Score`**: A crucial metric computed as $100 - 100 \cdot \frac{\text{Maximum Pareto Improvement}}{\text{Maximum Score per Player}}$. This score reflects how close the achieved agreement is to the optimal outcome, with higher scores indicating better performance relative to the maximum possible improvement.

