import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
import random
import uuid
from datetime import datetime, timedelta
from faker import Faker
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import warnings

fake = Faker('es_ES')


def generate_dni():
    POSSIBLE_LETTERS = (
        "T", "R", "W", "A", "G", "M", "Y", "F", "P", "D", "X", "B", "N", "J", "Z", "S", "Q", "V", "H", "L", "C", "K", "E", "T"
    )
    NUMBER_DNI = random.randint(10000000, 99999999)
    LETTER_DNI = POSSIBLE_LETTERS[NUMBER_DNI % 23]
    return f"{NUMBER_DNI}{LETTER_DNI}"


def generate_birthdate():
    age_ranges = [(18, 25), (26, 34), (35, 46), (47, 64), (65, 74), (75, 100)]
    probabilities = [0.2154, 0.2213, 0.2108, 0.1787, 0.1245, 0.049]
    selected_range = random.choices(age_ranges, probabilities)[0]
    age = random.randint(*selected_range)
    today = datetime.today()
    birthdate = today - timedelta(days=age * 365)
    return birthdate.date()


categorias_transaccion = {
    "Ocio y entretenimiento": {
        "subcategorias": ["cine", "concierto", "videojuego", "juego", "libreria", "museo"],
        "frecuencia": "semanal"
    },
    "Compras": {
        "subcategorias": ["ropa", "tienda", "electrodoméstico", "electronica"],
        "frecuencia": "semanal"
    },
    "Alimentacion": {
        "subcategorias": ["Supermercado", "Fruteria", "Charcuteria", "Ultramarinos", "Verduleria", "Panaderia", "Mercado"],
        "frecuencia": "diaria"
    },
    "Bares y Restaurantes": {
        "subcategorias": ["bar", "restaurante", "cafeteria", "bebida", "pintxos"],
        "frecuencia": "semanal"
    },
    "Transporte": {
        "subcategorias": ["gasolina", "taxi", "bus", "metro", "Peaje", "tarjeta de transporte"],
        "frecuencia": "diaria"
    },
    "Hogar y facturas": {
        "subcategorias": ["gas", "mantenimiento", "electricidad", "agua", 'hipoteca'],
        "frecuencia": "mensual"
    },
    "Salud y bienestar": {
        "subcategorias": ["Gimnasio", "Fisioterapia", "Farmacia"],
        "frecuencia": "mensual"
    },
    "Nominas": {
        "subcategorias": ["sueldo", "nómina"],
        "frecuencia": "mensual"
    },
    "Ingreso": {
        "subcategorias": ['transferencia', 'ingreso', 'ingreso en efectivo', 'bizum'],
        "frecuencia": "mensual"
    },
    'Cajero y transferencias': {
        "subcategorias": ['cajero', 'transferencia', "bizum"],
        "frecuencia": "semanal"
    },
    'Ahorro e inversión': {
        "subcategorias": ['plan de pensiones', 'fondos de inversión', 'acciones', 'Cuenta de ahorro'],
        "frecuencia": "mensual"
    },
    'Impuestos y tasas': {
        "subcategorias": ['IBI', 'IRPF', 'Hacienda'],
        "frecuencia": "anual"
    },
    'Seguros y finanzas': {
        "subcategorias": ['seguro de coche', 'seguro de hogar', 'Seguro de salud'],
        "frecuencia": "anual"
    },
    'Viajes y vacaciones': {
        "subcategorias": ['hotel', 'vuelo', 'agencia de viajes'],
        "frecuencia": "anual"
    },
    'Otros': {
        "subcategorias": ['varios', 'otros'],
        "frecuencia": "mensual"
    }
}


perfil_ingreso = {
    "Bajo": {
        "rango": (1000, 2000),
        "categorias_predominantes": ["Alimentacion", "Hogar y facturas", "Otros"],
        "frecuencia_ingreso": "Mensual",
        "probabilidad": 0.5  # 50% chance
    },
    "Medio": {
        "rango": (2001, 4000),
        "categorias_predominantes": ["Compras", "Alimentacion", "Bares y Restaurantes", "Ocio y entretenimiento"],
        "frecuencia_ingreso": "Mensual",
        "probabilidad": 0.4  # 40% chance
    },
    "Alto": {
        "rango": (4001, 10000),
        "categorias_predominantes": ["Viajes y vacaciones", "Ocio y entretenimiento", "Compras", "Salud y bienestar"],
        "frecuencia_ingreso": "Trimestral",
        "probabilidad": 0.1  # 10% chance
    }
}


def generate_user_profile():
    profiles = list(perfil_ingreso.keys())
    probabilities = [perfil_ingreso[profile]["probabilidad"] for profile in profiles]
    return random.choices(profiles, weights=probabilities)[0]

def generate_female_user():
    profile = generate_user_profile()
    return {
        "_id": str(uuid.uuid4()),
        "dni": generate_dni(),
        "name": fake.name_female(),
        "birthDate": generate_birthdate(),
        "city": fake.city(),
        "gender": "Female",
        "email": fake.free_email(),
        "password": fake.password(),
        "incomeProfile": profile,
        "incomeFrequency": perfil_ingreso[profile]["frecuencia_ingreso"]
    }

def generate_male_user():
    profile = generate_user_profile()
    return {
        "_id": str(uuid.uuid4()),
        "dni": generate_dni(),
        "name": fake.name_male(),
        "birthDate": generate_birthdate(),
        "city": fake.city(),
        "gender": "Male",
        "email": fake.free_email(),
        "password": fake.password(),
        "incomeProfile": profile,
        "incomeFrequency": perfil_ingreso[profile]["frecuencia_ingreso"]
    }


def generate_account(client_id):
    return {
        "clientId": client_id,
        "accountNumber": fake.iban(),
        "balance": round(random.uniform(100.0, 10000.0), 2),
        "transactions": []  
    }

def generate_realistic_amount(categoria, subcategoria, financial_profile):
    base_montos = {
        "Ocio y entretenimiento": (10, 100),
        "Compras": (20, 500),
        "Alimentacion": (5, 100),
        "Bares y Restaurantes": (5, 50),
        "Transporte": (1, 100),
        "Hogar y facturas": (50, 1000),
        "Salud y bienestar": (10, 200),
        "Nominas": (1000, 3000),
        "Ingreso": (100, 5000),
        'Ahorro e inversión': (50, 2000),
        'Impuestos y tasas': (100, 500),
        'Seguros y finanzas': (50, 400),
        'Viajes y vacaciones': (100, 2000),
        'Otros': (10, 500)
    }
    
    min_monto, max_monto = base_montos.get(categoria, (10, 500))
    
  
    if subcategoria in ['hipoteca', 'alquiler']:
        min_monto, max_monto = 500, 2000
    elif subcategoria in ['Supermercado', 'Mercado']:
        min_monto, max_monto = 30, 200
    elif subcategoria in ['gasolina', 'tarjeta de transporte']:
        min_monto, max_monto = 20, 80
    

    if financial_profile == "Bajo":
        max_monto *= 0.75
    elif financial_profile == "Alto":
        max_monto *= 1.5

    return round(np.random.normal(loc=(min_monto + max_monto) / 2, scale=(max_monto - min_monto) / 4), 2)


def generate_transaction_date(base_date, categoria):
    frecuencia = categorias_transaccion[categoria]['frecuencia']
    if frecuencia == 'diaria':
        days_to_add = random.randint(0, 30)
    elif frecuencia == 'semanal':
        days_to_add = random.randint(0, 7)
    elif frecuencia == 'mensual':
        days_to_add = random.randint(0, 3)
    elif frecuencia == 'anual':
        days_to_add = 0
    else:
        days_to_add = random.randint(0, 30)
    
    return base_date + timedelta(days=days_to_add)


def generate_transaction(client_id, account_number, financial_profile, base_date):
    categoria = random.choice(list(categorias_transaccion.keys()))
    subcategoria = random.choice(categorias_transaccion[categoria]['subcategorias'])
    
    monto = generate_realistic_amount(categoria, subcategoria, financial_profile)
    transaction_type = "Ingreso" if categoria in ["Nominas", "Ingreso"] else "Egreso"
    
    return {
        "clientId": client_id,
        "accountNumber": account_number,
        "amount": monto,
        "transactionType": transaction_type,
        "transactionDate": generate_transaction_date(base_date, categoria),
        "transactionCity": fake.city(),
        "transactionConcept": subcategoria,
        "transactionCategory": categoria,
        "transactionId": str(uuid.uuid4())
    }

def generate_transaction_history(client_id, account_number, financial_profile, num_months=4):
    history = []
    base_date = datetime.now() - timedelta(days=num_months * 30)
    
    for _ in range(num_months):
        unique_dates = set()
        num_transactions = random.randint(10, 20) if financial_profile == "Alto" else random.randint(2, 12)
        
        for _ in range(num_transactions):
            while True:
                transaction_date = base_date + timedelta(days=random.randint(1, 30))
                if transaction_date not in unique_dates:
                    unique_dates.add(transaction_date)
                    break
                
        for _ in range(num_transactions):
            transaction = generate_transaction(client_id, account_number, financial_profile, base_date)
            history.append(transaction)
        
        income_transaction = generate_transaction(client_id, account_number, financial_profile, base_date)
        income_transaction["transactionType"] = "Ingreso"
        income_transaction["transactionCategory"] = "Nominas"
        income_transaction["transactionConcept"] = "sueldo"
        income_transaction["amount"] = generate_realistic_amount("Nominas", "sueldo", financial_profile)
        history.append(income_transaction)
        
        base_date += timedelta(days=30)
    
    return sorted(history, key=lambda x: x['transactionDate'])

user = generate_female_user()
account = generate_account(user["_id"])
account["transactions"] = generate_transaction_history(user["_id"], account["accountNumber"], user["incomeProfile"])

print("Usuario generado:", user)
print("Cuenta generada:", account)
print(f"Número total de transacciones: {len(account['transactions'])}")

def generate_multiple_users_and_transactions(num_users=10, num_months=4):
    all_users = []
    all_accounts = []
    all_transactions = []

    for _ in range(num_users):
        user = generate_female_user() if random.choice([True, False]) else generate_male_user()
        account = generate_account(user["_id"])
        transactions = generate_transaction_history(user["_id"], account["accountNumber"], user["incomeProfile"], num_months)
        
        all_users.append(user)
        all_accounts.append(account)
        all_transactions.extend(transactions)

    return all_users, all_accounts, all_transactions

def convert_to_dataframes(users, account, transactions):
    users_df = pd.DataFrame(users)
    account_df = pd.DataFrame(account)
    transactions_df = pd.DataFrame(transactions)
    
    # Convertir fechas a datetime
    users_df['birthDate'] = pd.to_datetime(users_df['birthDate'])
    transactions_df['transactionDate'] = pd.to_datetime(transactions_df['transactionDate'])
    
    return users_df, account_df, transactions_df

num_users = 100
num_months = 4
users, accounts, transactions = generate_multiple_users_and_transactions(num_users, num_months)
users_df, accounts_df, transactions_df = convert_to_dataframes(users, accounts, transactions)
merged_users_accounts_df = pd.merge(users_df, accounts_df, left_on='_id', right_on='clientId', how='inner')
final_df = pd.merge(merged_users_accounts_df, transactions_df, on='accountNumber', how='inner')

users=final_df[['_id', 'dni', 'name', 'birthDate', 'city', 'gender', 'email',
       'password', 'incomeProfile', 'incomeFrequency']].copy()
accounts=final_df[['clientId_x',
       'accountNumber', 'balance', 'transactions']].copy()
transactions=final_df[['clientId_y', 'amount',
       'transactionType', 'transactionDate', 'transactionCity',
       'transactionConcept', 'transactionCategory', 'transactionId']].copy()
seed=42
X_train, X_test, y_train, y_test = train_test_split(transactions.drop(["transactionCategory","clientId_y"],axis=1),
                                                    transactions["transactionCategory"],
                                                    test_size=0.25,
                                                    random_state=seed)


transactions['transactionConcept'] = transactions['transactionConcept'].str.lower().str.replace(r'[^\w\s]', '', regex=True)
vectorizer = TfidfVectorizer(max_features=1000)
X_text = vectorizer.fit_transform(transactions['transactionConcept'])
y_category = transactions['transactionCategory']
X_train, X_test, y_train, y_test = train_test_split(X_text, y_category, test_size=0.3, random_state=42)

forest_clf_param={"n_estimators":[5,10,25],
                "random_state":[seed],
                "max_depth":[1,2,3],
                "criterion":["gini", "entropy", "log_loss"],
                "bootstrap":[True,False],
                "min_samples_split":[2,3,5],
                "min_samples_leaf":[1,3,5],
                "min_weight_fraction_leaf":[0.0,0.25],
                "class_weight":[None,"balanced"],
                "ccp_alpha":[0.0,0.25],
                "warm_start":[True,False]}

forest_clf_puro= RandomForestClassifier()
forest_clf_puro=GridSearchCV(forest_clf_puro,forest_clf_param,refit=True)
warnings.filterwarnings(action='ignore')
forest_clf_puro.fit(X_train, y_train)
model=forest_clf_puro.best_estimator_

app = FastAPI()
categorias = [ "Ocio y entretenimiento","Compras","Alimentacion","Bares y Restaurantes","Transporte","Hogar y facturas","Salud y bienestar","Nominas",'Ingreso','Cajero y transferencias','Ahorro e inversión','Impuestos y tasas','Seguros y finanzas','Viajes y vacaciones','Otros']

class InputData(BaseModel):
    data:list
    shape:list

#def predict_inputdata(data: InputData):
        #sparse_matrix = csr_matrix((data.data, (range(len(data.data)), data.shape)), shape=data.shape)
        #prediction = model.predict(X_test)
        #prediction = categorias[int(prediction[0])]
        #return prediction


@app.get("/")
def read_root():
    return {"Message": "Esto es una API de un modelo (todavía en desarrollo) que permite categorizar los gastos e ingresos de una serie de transacciones",
            "Categorias (respuesta esperada)": categorias}

@app.get("/DARLE AQUI, PARA QUE MUESTRE CATEGORIAS DE DATOS CREADOS EN LA PROPIA API (Datos Train)")
def predict_funcional_train():
        prediction = model.predict(X_train)
        prediction_str = ', '.join(map(str, prediction))
        return {"Predicciones": prediction_str}

@app.post("/predict con input data (en progreso)")
def features():
    """_Resumen_

    Esto es una API de un modelo (todavía en desarrollo) que permite categorizar los gastos e ingresos de una serie de transacciones.
    
    Args: 
    Esto funciona por matriz, por lo tanto, se requiere los datos que contiene la matriz 
    y la forma de la matriz. La API ya fabrica sus propios datos porque no hay tiempo, pero en principio,
    esto debería poder darte algo si pusieras...
    Algo así:
    {
    "data": [0.1, 0.2, 0.3],
    "shape": [3, 1]
    }

    data: [datos]

    shape: [nº de datos en cada fila, nº de filas]

    Returns:
        __type__:__description__
    """