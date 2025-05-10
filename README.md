

## 🌆 Urban Climate Risk Predictor 2030

A **Python-based web application** that predicts the level of urban climate risk (Low, Medium, High) for different environments projected for the year 2030. It uses machine learning to evaluate environmental and infrastructural factors to help cities prepare for climate-related challenges.

---

### 📌 Features

* Predicts **urban climate risk** based on temperature, humidity, wind speed, precipitation, urbanization, and green space.
* Interactive **web interface** built with Flask.
* Trained with real or simulated environmental data.
* Styled with `Bootstrap` and custom CSS.
* Interpretable output with graphical visualizations.
* Scalable for integration with GIS or policy dashboards.

---

### 🧠 Tech Stack

| Component              | Description                     |
| ---------------------- | ------------------------------- |
| `Flask`                | Web framework for the app       |
| `scikit-learn`         | Machine Learning model training |
| `pandas`/`numpy`       | Data manipulation               |
| `matplotlib`/`seaborn` | Visualizations                  |
| `HTML/CSS`             | Frontend form & result display  |

---

### 📁 File Structure

```
UrbanClimateRiskPredictor2030/
├── app.py                   # Flask app entry point
├── main.py                 # Runs the app
├── models.py               # Model training, saving, and loading
├── prediction_logic.py     # Core prediction logic
├── train.ipynb             # Model training Jupyter notebook
├── utils.py                # Helper functions for encoding, scaling
├── urbanclimateriskpredictor2030.csv  # Dataset
├── static/
│   └── custom.css          # Custom styles
├── templates/
│   ├── index.html          # Main user interface
│   └── templates.html      # Optional base layout
├── model.pkl               # Trained ML model
├── scaler.pkl              # StandardScaler object
├── label_encoder.pkl       # LabelEncoder object
└── requirements.txt        # Python dependencies
```

---

### 🧪 Model Inputs

| Feature       | Type                                | Example |
| ------------- | ----------------------------------- | ------- |
| Temperature   | Float                               | 30.5 °C |
| Precipitation | Float                               | 120 mm  |
| Humidity      | Float                               | 80%     |
| Wind Speed    | Float                               | 12 km/h |
| Urban Area    | Categorical (`Urban`/`Rural`)       |         |
| Green Space   | Categorical (`Low`/`Medium`/`High`) |         |

---

### 🧾 Installation Steps

1. **Clone the Repository**

```bash
git clone https://github.com/yourusername/urban-climate-risk-predictor-2030.git
cd urban-climate-risk-predictor-2030
```

2. **Create a Virtual Environment**

```bash
python -m venv venv
source venv/bin/activate     # On Windows: venv\\Scripts\\activate
```

3. **Install Dependencies**

```bash
pip install -r requirements.txt
```

4. **Run the App**

```bash
python main.py
```

Visit [http://localhost:5000](http://localhost:5000) to use the predictor.

---

### 📊 Sample Output

* Predicted Risk Level: **High**
* Interpretation: "This region is likely to face high climate risk in 2030. Sustainable infrastructure and increased green space are recommended."

---

### 📉 Visualizations (Generated from `train.ipynb`)

* `risk_by_green_space.png`: Average risk per green space category
* `heatmap_temp_humidity.png`: Risk vs. temperature & humidity
* `risk_scatter.png`: Wind speed vs. precipitation vs. risk

---

### 💬 How to Use

1. Open the form at `/` in your browser.
2. Enter environmental parameters.
3. Submit the form to get the risk prediction and interpretation.

---

### ✅ Ideal For

* Urban Planners
* Environmental Analysts
* Students/Researchers in Climate Science
* Smart City Developers

---

### 💡 Future Scope

* Add map-based input using Leaflet.js or Google Maps API
* API endpoint for mobile apps or dashboards
* Risk classification heatmaps for multiple cities

---


