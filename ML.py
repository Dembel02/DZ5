# Загрузка библиотек
import pandas as pd
import random
import numpy as np

from sklearn.preprocessing import LabelEncoder

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

from sklearn.model_selection import train_test_split

from catboost import CatBoostClassifier

from sklearn.metrics import accuracy_score

from flask import Flask, request, jsonify

app = Flask(__name__)

# Набор констант по программе
MODEL_FILE_EFF = 'people_university_model_eff.cbm'
MODEL_FILE = 'people_university_model.cbm'
DF_COUNT = 2001
SUBJ_COUNT = 6
MIN_GRADE = 40
MAX_GRADE = 99
FINAL_RESULT_LIST = ['удовлетворительно', 'хорошо', 'отлично']

# Набор def и классов


# 1. Входные данные (загрузка/генерация)
def generate_student():
    '''
    Создайте датасет из не менее чем 2000 записей, содержащий датасет о
    среднем балле студентов (от 0 до 99) по 6 предметам и оценка итоговой
    лабароторной работы (удовлетворительно, хорошо, отлично).
    Названия предметов, средний балл, оценка итоговой лабароторной работы задается про
    на ваше усмотрение.
    '''
    def generate_final (row):
        # Балл ниже 70 - удовл
        cond_c = row['mean_score'] < 70
        # Балл ниже 85 - хор
        cond_b = row['mean_score'] >= 70
        cond_b1 = row['mean_score'] < 85
        # Балл выше 85 - отл
        cond_a = row['mean_score'] >= 85
        if(cond_c):
            val = FINAL_RESULT_LIST[0]
        elif(cond_b and cond_b1):
            val = FINAL_RESULT_LIST[1]
        else:
            val = FINAL_RESULT_LIST[2]
        return val
    subjects = [f"subject_{num + 1}" for num in range(SUBJ_COUNT)]
    grades = np.random.randint(MIN_GRADE, MAX_GRADE + 1, (DF_COUNT, SUBJ_COUNT))
    students_scores = pd.DataFrame(grades, columns = subjects)

    students_scores.sum(axis = 1)
    students_scores['mean_score'] = np.round((students_scores.sum(axis = 1)/SUBJ_COUNT),)
    print(students_scores.head())

    students_scores['final_lab'] = students_scores.apply(generate_final, axis=1)
    print(students_scores.head())
    return students_scores


def classifyStudent(new_student):
    '''
    Функция для классификации студентов
    :param new_student: массив ключ-значение или строка в датасете
    :return:
    '''
    # Чтение обученной модели из файла
    model = CatBoostClassifier()
    model.load_model(MODEL_FILE)
    # Преобразование данных студента в DataFrame
    new_data = pd.DataFrame([new_student])

    # Использование модели для предсказания
    predicted_category = model.predict(new_data)[0]

    # Преобразование обратно в текстовую категорию
    categories = [...] # Категориальная по студентам
    predicted_category = categories[predicted_category]

    return predicted_category

#df = generate_student()
#df.to_csv('student.csv', index=False)

df = pd.read_csv('student.csv')
X = df.drop(['final_lab', 'mean_score'], axis=1)
print(X)
y = df['final_lab']
print(y)
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)
print(y)

# 2. Выберите лучшие 3 признака для обучения
selector = SelectKBest(chi2, k=2)
X_new = selector.fit_transform(X, y)

# 3. Разбейте датасет на тестовую и обучающую выборку train_test_split (для лучших признаков)
# 80% обучающей выборки, 20% тестовой
X_train_eff, X_test_eff, y_train_eff, y_test_eff = train_test_split(X_new, y, test_size=0.2, random_state=1818)

# 3.1* Разбейте датасет на тестовую и обучающую выборку train_test_split (для всех признаков)
# 80%  обучающей выборки, 20% тестовой
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

# 4. Обучаем модель CatBoostClassyfier
model_eff = CatBoostClassifier(iterations=100, depth=4, learning_rate=0.1)
model_eff.fit(X_train_eff, y_train_eff)
# Сохранение модели в файл
model_eff.save_model(MODEL_FILE_EFF)

model = CatBoostClassifier(iterations=100, depth=4, learning_rate=0.1)
model.fit(X_train, y_train)

# Сохранение модели в файл
model.save_model(MODEL_FILE)

# Проводим тестирование модели, результатом является точность модели (accurasy)
y_pred_eff = model_eff.predict(X_test_eff)

y_pred = model.predict(X_test)

# Оцениваем точность
accuracy_eff = accuracy_score(y_test_eff, y_pred_eff)
print(f"точность модели: {accuracy_eff}")

accuracy = accuracy_score(y_test, y_pred)
print(f"точность модели: {accuracy}")

def load_model():
    global model
    try:
        model = CatBoostClassifier()
        model.load_model('people_university_model.cbm')
    except:
        trainModel()
        model = CatBoostClassifier()
        model.load_model('people_university_model.cbm')

# Метод для обучения модели
@app.route('/train', methods=['GET'])
def train():
    try:
#        createFrame()  # Создание данных для обучения (может быть переделано для загрузки из запроса)
        trainModel()  # Обучение модели
        return jsonify({'message': 'model saved'})
    except Exception as e:
        return jsonify({'error': str(e)})

# Метод для классификации новой квартиры
@app.route('/classify', methods=['GET'])
def classify():
    try:
        # Получение параметров из GET-запроса
        subject_1 = request.args.get('subject_1')
        subject_2 = request.args.get('subject_2')
        subject_3 = request.args.get('subject_3')
        subject_4 = request.args.get('subject_4')
        subject_5 = request.args.get('subject_5')
        subject_6 = request.args.get('subject_6')


        # Прогнозирование с использованием обученной модели
        predicted_category = model.predict([[
            subject_1, subject_2, subject_3, subject_4, subject_5, subject_6
        ]])[0]

        # Преобразование результатов в текстовую категорию
        categories = ['удовлетворительно', 'хорошо', 'отлично']
        predicted_category = categories[predicted_category]
        return f"Category: {predicted_category}"
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    load_model()
    app.run(debug=True)