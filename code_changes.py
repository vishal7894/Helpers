from datetime import datetime

today = datetime.today().strftime('%Y-%m-%d')
df = df[df["ReferenceDate"]<=today]

