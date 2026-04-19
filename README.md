# Tour Satisfaction Predictor

This project implements a K-Nearest Neighbors (KNN) model to predict customer satisfaction based on tour data. The model is deployed as a Streamlit web application.

## Project Structure
- `app.py`: The Streamlit web application.
- `scaler.pkl`: The StandardScaler object used for numerical feature scaling.
- `encoder.pkl`: The OneHotEncoder object used for categorical feature encoding.
- `knn_model.pkl`: The trained K-Nearest Neighbors model.
- `cleaned_tour_data.csv`: The cleaned dataset used for training and for populating dropdowns in the Streamlit app.
- `requirements.txt`: A list of Python dependencies.
- `README.md`: This file.

## Setup and Installation (Local System)

### 1. Download Files
Download the following files from your Colab environment (or wherever they are stored) and place them all in the same directory on your local machine:
- `app.py`
- `scaler.pkl`
- `encoder.pkl`
- `knn_model.pkl`
- `cleaned_tour_data.csv`
- `requirements.txt` (generated in Colab)
- `README.md` (generated in Colab)

### 2. Install Python Dependencies
Open your terminal or command prompt, navigate to the directory where you saved the files, and install the required Python packages using `pip`:

```bash
pip install -r requirements.txt
```

### 3. Running the Streamlit Application
Once all dependencies are installed, you can launch the Streamlit application from your terminal:

```bash
streamlit run app.py
```

This command will open a new tab in your web browser displaying the Streamlit application. If it doesn't open automatically, look for a local URL (e.g., `http://localhost:8501`) printed in your terminal and open it manually.

### (Optional) Exposing your local app with `ngrok`
If you want to share your locally running Streamlit app with others over the internet, you can use `ngrok`.

1.  **Download ngrok**: Download and install `ngrok` from the official website: [ngrok.com/download](https://ngrok.com/download).
2.  **Authenticate ngrok**: Set up your authentication token (obtained from your ngrok dashboard) in your terminal:
    ```bash
    ngrok authtoken <YOUR_AUTHTOKEN>
    ```
3.  **Run ngrok**: With your Streamlit app running (from `streamlit run app.py`), open another terminal window and run ngrok to tunnel port 8501:
    ```bash
    ngrok http 8501
    ```
    `ngrok` will provide a public URL that you can share.
