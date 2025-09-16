# Face-Emotion-detection-transfer-learning
This project implements a Face Emotion Detection system using Python, PyTorch, and transfer learning with a pre-trained ResNet18 model. The system classifies facial expressions into various emotional categories (e.g., happy, sad, angry, surprised, etc.) by leveraging the feature extraction capabilities of ResNet18, fine-tuned on a labeled facial emotion dataset. The pipeline includes data preprocessing, model adaptation, training, and evaluation, enabling real-time or image-based emotion recognition with high accuracy and efficiency.

## Team Members
| AC.NO    | Name                   | Role           | Contributions                           |
|----------|------------------------|----------------|-----------------------------------------|
| 202270049| Yousif Adel Al-Hashedy | Lead Developer | Project File Preperation, GUI developer |
| 202270041| Abdulkhaliq Mohammed Al-Mohammedi | esp32 programmeing, Mqtt |
| 202270081| Hussam Waleed Al-Shamaj | DL Engineer | Model optimization, evaluation metrics |

## Installation and Setup

### Prerequisites
- Python 3.13.5 (specified in `.python-version`)
- UV package manager

### Installation Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/Jousifhome/Emotion-detection-transfer-learning.git
   cd Emotion-detection-transfer-learning
   ```

2. Install dependencies using UV:
   ```bash
   uv sync
   ```

3. Run the project:
   ```bash
   uv run python main.py
   uv run python 
   ```

## Project Structure

```
Emotion-detection-transfer-learning/
├── README.md              # Project documentation
├── pyproject.toml         # UV project configuration
├── .python-version        # Python version specification
|── ModelTraining Pytorch.ipynb
├── main.py               # Main application entry point
├── src/                  # Source code
│   ├── data/            # Data processing modules
│   ├── models/          # ML model implementations
│   └── utils/           # Utility functions
├── notebooks/           # Jupyter notebooks
├── data/               # Dataset files
└── docs/               # Additional documentation
```
## Usage
### Running Experiments
```bash
# Run the main application
uv run python main.py

# Run the training script
uv run python src/models/TraingModelCode.py

# Run the test suite
uv run python test_project.py

# Test module imports
uv run python -c "from src.models import SpamClassifier; print('Model imported successfully')"

# Run Jupyter notebook
uv run jupyter notebook  

```

## Results

- **Model Accuracy**: 96.16%
- **Training Time**: 2.3 minutes
- **Algorithm**: CNN (ResNet18 pertrained model)
- **Key Findings**: 
  - The training accuracy found to be 96.16% 
  - The testing accuracy is found to be 65.35%
  -  Per-Class Accuracy:
        angry          : 58.66%
        disgust        : 52.25%
        fear           : 41.41%
        happy          : 86.58%
        neutral        : 61.15%
        sad            : 56.86%
        surprise       : 77.98%
  - Model achieves high precision and recall be using opencv with haarcascade_frontalface_default.xml



