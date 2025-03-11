# StataAgent

StataAgent is an AI-powered agent designed for intuitive, question-based data exploration and analysis directly within Stata. Built upon the smolagents framework, StataAgent interprets both simple Stata commands and more complex analytical questions, converting them into executable Stata code. With an integrated understanding of dataset metadata—including variable names and labels—StataAgent streamlines analytical workflows, making data exploration seamless and efficient.

---


***As of 11 March 2025: this is an experimental project***


## Features

- **Natural Language Queries:**
  - Easily handle questions like:
    - "What is the effect of x1 on y, controlling for x2?"
    - "What is the distribution of homeowners over time?"

- **Command Execution:**
  - Directly execute straightforward Stata commands like `regress y on x1 and x2`.

- **Metadata Integration:**
  - Automatically leverages dataset metadata to accurately interpret questions and commands.

- **Powered by smolagents:**
  - Robust AI framework optimized for minimal overhead and maximal efficiency.

---

## Installation

### Prerequisites

- Stata (version 15 or newer recommended)
- Python 3.7 or newer
- Hugging Face `transformers` library and API key

### Setup
Clone this repository and navigate to the project directory:

```bash
git clone https://github.com/yourusername/StataAgent.git
cd StataAgent
```

Set up a Python virtual environment (recommended):

```bash
python -m venv venv
source venv/bin/activate  # or on Windows: venv\Scripts\activate
```

Install required Python packages:

```bash
pip install -r requirements.txt
```

Ensure `smolagents` is properly installed and configured per [smolagents documentation](https://github.com/smol-ai/smolagents).

---

## Usage

Launch StataAgent with:

```bash
python main.py
```

Once running, interact with the agent using plain English queries or standard Stata commands.

### Examples

**Natural Language Query:**

```
What is the effect of education on wages, controlling for experience and gender?
```

**Internally executed Stata Command:**

```stata
regress wage education experience gender
```

**Natural Language Query:**

```
What is the distribution of homeowners by year?
```

**Internally Executed Stata Command:**

```stata
tab homeowners year
```

The results of the executed command will be displayed in the console.



** Note: As of now there is no way to retain data in memory, so each query loads the data from disk.
As this is not computationally efficient, StataAgent is not recommended for datasets >100,000 observations.

---

## Contributing

Contributions are welcome! Please:

1. Fork this repository.
2. Create a feature branch (`git checkout -b feature/your-feature`).
3. Commit your changes (`git commit -am 'Add feature'`).
4. Push to the branch (`git push origin feature/your-feature`).
5. Submit a Pull Request.

---


## Acknowledgments

- [smolagents](https://github.com/smol-ai/smolagents): Lightweight AI agent framework.
- StataCorp for providing robust statistical analysis software.

---

**Happy exploring with StataAgent!**

