# Car Damage Detection

> [!NOTE]
> Archived university project for Digital Image Processing at Telkom University (2024). The README was updated in 2026 for archiving purposes. The code and model are as submitted.

Car Damage Detection is a desktop application that uses a trained Keras model to classify vehicle damage in a photograph as minor, moderate or severe. The application was developed in 2024 for the Digital Image Processing course at Telkom University.

<p align="center">
  <img src="https://github.com/raflymg02/Car-Damage-Detection/assets/120699410/14329716-8156-49cc-abbc-b4d0ac2b30c6" alt="Application main screen" height="150">
  <img src="https://github.com/raflymg02/Car-Damage-Detection/assets/120699410/582f5cf4-2fc1-4a93-a523-32f102eb61f7" alt="Minor damage classification result" height="150">
  <img src="https://github.com/raflymg02/Car-Damage-Detection/assets/120699410/e39478d8-648a-4da3-a186-17ee39cfb6f7" alt="Moderate damage classification result" height="150">
  <img src="https://github.com/raflymg02/Car-Damage-Detection/assets/120699410/17666286-dfa6-4991-a01e-fd7d361de39e" alt="Severe damage classification result" height="150">
</p>

## Running the application

### Windows build

Download `Car-Damage-Detection.rar` from the [GitHub Releases page](https://github.com/raflymg02/Car-Damage-Detection/releases) and follow the instructions in the release notes.

### Run from source

Requires Python 3.9–3.12. Run these commands in Windows Command Prompt.

1. Clone the repository and open the project directory.

   ```cmd
   git clone https://github.com/raflymg02/Car-Damage-Detection.git
   cd Car-Damage-Detection
   ```

2. Create and activate a virtual environment (optional, but recommended).

   ```cmd
   python -m venv .venv
   .venv\Scripts\activate
   ```

3. Install the dependencies.

   ```cmd
   python -m pip install tensorflow==2.16.1 keras==3.3.3 numpy pillow matplotlib
   ```

4. Run the application from the project directory.

   ```cmd
   python main.py
   ```

If using a virtual environment, activate it again when running the application from a new Command Prompt session.

## Dataset

The model was trained on the Car Damage Severity Dataset [1]. The dataset is licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/), and the bundled model is shared under the same licence.

[1] P. Bhamere, 'Car Damage Severity Dataset', Kaggle. [Online]. Available: https://www.kaggle.com/datasets/prajwalbhamere/car-damage-severity-dataset

## Contributors

- Rafly Muhammad Ghifari
- Yuridikta Adha Muslim
- Angly Khan Surya
