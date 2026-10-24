# Hostel Mess Food-Waste Decider (beginner steps, Windows)

1. Install Python 3.10+ from python.org (tick "Add Python to PATH").
2. Unzip this folder. Open it, click the address bar, type `cmd`, press Enter.
3. Create environment:   python -m venv venv
4. Activate it:          venv\Scripts\activate        (Mac/Linux: source venv/bin/activate)
5. Install packages:     pip install -r requirements.txt
6. (Optional) new data:  python generate_data.py      (data/mess_data.csv is already included)
7. Train the model:      python train.py
8. See results:          python evaluate.py           (charts saved in results/)
9. Compare models:       python compare_models.py
10. Run the dashboard:   streamlit run app.py         (opens http://localhost:8501)

Real data: fill data/real_data_template.csv format, save as data/mess_data.csv, repeat steps 7-10.

Deploy: push folder to GitHub -> share.streamlit.io -> New app -> main file app.py -> Deploy.
