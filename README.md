
# Agrimetaverse-Intelligence

A next-generation AI platform using **SARIMAX** to forecast prices of 22 agricultural commodities, enabling smarter decision-making for farmers and traders.

## 🚀 Features

- Accurate price forecasting for 22 agricultural commodities
- SARIMAX time series modeling
- Interactive data visualization and historical trend analysis
- Actionable insights for farmers, traders, and policymakers

## 📊 Example Commodities
- Rice, Wheat, Maize, Pulses, Sugarcane, Cotton, Oilseeds, Fruits, Vegetables, Spices, and more

## 🛠️ Installation

```bash
git clone https://github.com/Bhuvaneshpree/Agrimetaverse-Intelligence.git
cd Agrimetaverse-Intelligence
pip install -r requirements.txt
```

## 🧑‍💻 Usage

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the primary prediction script (monthly data):
```bash
streamlit run AgriPredict.py
```

3. Alternatively, run the annual data prediction script:
```bash
streamlit run agripricepredictor.py
```

Note: Both scripts use Streamlit for the interactive interface. The scripts will automatically load the appropriate datasets and provide a web interface for commodity selection and forecasting.

## 📁 Project Structure

```
Agrimetaverse-Intelligence/
├── AgriPredict.py              # Primary SARIMAX forecasting script
├── agripricepredictor.py       # Alternative prediction script
├── datamain.csv                # Main dataset (monthly data)
├── DatasetSIH1647.csv          # Annual aggregated dataset
├── requirements.txt            # Python dependencies
├── LICENSE                     # MIT License
├── .gitignore                  # Git ignore rules
└── README.md                   # Project documentation
```

## 🤝 Contributing

Contributions, suggestions, and feature requests are welcome! Please open an issue or submit a pull request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

*Empowering the agricultural sector with AI-driven insight!*
