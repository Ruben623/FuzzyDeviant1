import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from skfuzzy import gaussmf

def zmf(x, a, b):
    """Z-образная функция принадлежности (для low)"""
    y = np.zeros_like(x)
    idx = np.where(x <= a)
    y[idx] = 1
    idx = np.where((a < x) & (x < b))
    y[idx] = 1 - (x[idx] - a) / (b - a)
    return y

def smf(x, a, b):
    """S-образная функция принадлежности (для high)"""
    y = np.zeros_like(x)
    idx = np.where(x >= b)
    y[idx] = 1
    idx = np.where((a < x) & (x < b))
    y[idx] = (x[idx] - a) / (b - a)
    return y

def setup_fuzzy_system():

    # Универсум от 0 до 10 с шагом 0.1 для плавности
    universe = np.arange(0, 10.1, 0.1)

    # Входные переменные
    aggression = ctrl.Antecedent(universe, 'aggression')
    delinquency = ctrl.Antecedent(universe, 'delinquency')
    norms = ctrl.Antecedent(universe, 'norms')
    autoaggression = ctrl.Antecedent(universe, 'autoaggression')
    manipulation = ctrl.Antecedent(universe, 'manipulation')

    # Настраиваем функции принадлежности
    for var in [aggression, delinquency, norms, autoaggression, manipulation]:

        # Low (Z-образная функция)
        var['low'] = fuzz.zmf(var.universe, 2, 4)
        # Medium (Гауссова функция с центром в 5)
        var['medium'] = fuzz.gaussmf(var.universe, 5, 1.2)
        # High (S-образная функция)
        var['high'] = fuzz.smf(var.universe, 6, 8)

        # Настройка автоматического отображения для проверки
        var.view()

    # Выходная переменная
    deviance = ctrl.Consequent(universe, 'deviance')

    # Настройка функций принадлежности для выходной переменной
    deviance['low'] = fuzz.zmf(deviance.universe, 1.5, 3.5)
    deviance['medium'] = fuzz.gaussmf(deviance.universe, 5, 1.5)
    deviance['high'] = fuzz.smf(deviance.universe, 6.5, 8.5)

    # Визуализация для проверки
    deviance.view()

    # Полный набор правил
    rules = [
    # Девиантность явно выражена
    # Доминирование высоких показателей
    ctrl.Rule(aggression['high'] & delinquency['high'], deviance['high']),
    ctrl.Rule(aggression['high'] & norms['high'], deviance['high']),
    ctrl.Rule(aggression['high'] & autoaggression['high'], deviance['high']),
    ctrl.Rule(aggression['high'] & manipulation['high'], deviance['high']),
    ctrl.Rule(delinquency['high'] & norms['high'], deviance['high']),
    ctrl.Rule(delinquency['high'] & autoaggression['high'], deviance['high']),
    ctrl.Rule(delinquency['high'] & manipulation['high'], deviance['high']),
    ctrl.Rule(norms['high'] & autoaggression['high'], deviance['high']),
    ctrl.Rule(norms['high'] & manipulation['high'], deviance['high']),
    ctrl.Rule(autoaggression['high'] & manipulation['high'], deviance['high']),

    # Критические комбинации с аутоагрессией
    ctrl.Rule(autoaggression['high'] & aggression['medium'], deviance['high']),
    ctrl.Rule(autoaggression['high'] & delinquency['medium'], deviance['high']),
    ctrl.Rule(autoaggression['high'] & norms['high'], deviance['high']),

    # Агрессия + манипуляции
    ctrl.Rule(aggression['high'] & manipulation['medium'], deviance['high']),
    ctrl.Rule(aggression['medium'] & manipulation['high'], deviance['high']),

    # Тройные комбинации
    ctrl.Rule(aggression['high'] & delinquency['high'] & norms['medium'], deviance['high']),
    ctrl.Rule(aggression['high'] & manipulation['high'] & autoaggression['medium'], deviance['high']),

    # Умеренная девиантность
    # Парные средние уровни
    ctrl.Rule(aggression['medium'] & delinquency['medium'], deviance['medium']),
    ctrl.Rule(aggression['medium'] & norms['medium'], deviance['medium']),
    ctrl.Rule(aggression['medium'] & manipulation['medium'], deviance['medium']),
    ctrl.Rule(delinquency['medium'] & norms['medium'], deviance['medium']),

    # Один высокий показатель
    ctrl.Rule(aggression['high'] & delinquency['low'] & norms['low'], deviance['medium']),
    ctrl.Rule(delinquency['high'] & manipulation['low'] & autoaggression['low'], deviance['medium']),

    # Комбинации с аутоагрессией
    ctrl.Rule(autoaggression['medium'] & aggression['low'], deviance['medium']),
    ctrl.Rule(autoaggression['medium'] & delinquency['low'], deviance['medium']),

    # Нормальное поведение
    ctrl.Rule(aggression['low'] & delinquency['low'] & norms['low'] & autoaggression['low'] & manipulation['low'], deviance['low']),

    ctrl.Rule(aggression['low'] & delinquency['low'] & norms['medium'], deviance['low']),
    ctrl.Rule(aggression['low'] & manipulation['low'] & autoaggression['medium'], deviance['low']),

    ctrl.Rule(autoaggression['high'] & aggression['low'] & delinquency['low'], deviance['low']),

    # Крайние случаи
    # Противоречивые комбинации
    ctrl.Rule(autoaggression['high'] & aggression['low'], deviance['medium']),
    ctrl.Rule(manipulation['high'] & delinquency['low'], deviance['medium']),

    # Приоритетные правила
    ctrl.Rule(
        (aggression['high'] | delinquency['high']) &
        (norms['high'] | manipulation['high']) &
        (autoaggression['medium'] | autoaggression['high']),
        deviance['high']
    )
]

    # Создание системы
    deviance_ctrl = ctrl.ControlSystem(rules)
    return ctrl.ControlSystemSimulation(deviance_ctrl)

def create_membership_plots():
    # Создаем фигуру с 6 субплoтами (5 входных + 1 выходная переменная)
    fig, axes = plt.subplots(nrows=6, ncols=1, figsize=(10, 15))
    plt.subplots_adjust(hspace=0.5)  # Добавляем пространство между графиками

    # Универсальный универсум для всех переменных [0, 10]
    universe = np.arange(0, 10.1, 0.1)

    # Настройка входных переменных
    input_vars = {
        'aggression': 'Агрессивное поведение',
        'delinquency': 'Противоправное поведение',
        'norms': 'Игнорирование норм',
        'autoaggression': 'Аутоагрессия',
        'manipulation': 'Манипуляции'
    }

    # Общие параметры для входных переменных
    params = {
        'low': {'type': 'zmf', 'a': 2, 'b': 4},
        'medium': {'type': 'gaussmf', 'mean': 5, 'sigma': 1.2},
        'high': {'type': 'smf', 'a': 6, 'b': 8}
    }

    for idx, (var_name, title) in enumerate(input_vars.items()):
        # Создаем антецедент
        var = ctrl.Antecedent(universe, var_name)
        
        # Настраиваем функции принадлежности
        var['low'] = fuzz.zmf(var.universe, params['low']['a'], params['low']['b'])
        var['medium'] = fuzz.gaussmf(var.universe, params['medium']['mean'], params['medium']['sigma'])
        var['high'] = fuzz.smf(var.universe, params['high']['a'], params['high']['b'])
        
        # Визуализация
        var.view(ax=axes[idx])
        axes[idx].set_title(title, pad=20, fontsize=12)
        axes[idx].set_ylabel('Принадлежность', fontsize=10)
        axes[idx].set_xlabel('Уровень', fontsize=10)
        axes[idx].legend(loc='upper right')

    # Настройка выходной переменной (девиантность)
    deviance = ctrl.Consequent(universe, 'deviance')
    
    # Специальные параметры для выходной переменной
    deviance['low'] = fuzz.zmf(deviance.universe, 1.5, 3.5)
    deviance['medium'] = fuzz.gaussmf(deviance.universe, 5, 1.5)
    deviance['high'] = fuzz.smf(deviance.universe, 6.5, 8.5)
    
    # Визуализация
    deviance.view(ax=axes[5])
    axes[5].set_title('Общая девиантность', pad=20, fontsize=12)
    axes[5].set_ylabel('Принадлежность', fontsize=10)
    axes[5].set_xlabel('Уровень', fontsize=10)
    axes[5].legend(loc='upper right')

    # Настройка внешнего вида
    plt.tight_layout()
    
    # Конвертируем в base64
    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    membership_plot = base64.b64encode(img.getvalue()).decode()
    plt.close()
    
    return membership_plot

def calculate_deviance(scores, max_scores):
    # Нормализация баллов к шкале 0-10
    normalized_scores = {
        'aggression': (scores['Агрессивное и рискованное поведение'] / max_scores['Агрессивное и рискованное поведение']) * 10,
        'delinquency': (scores['Делинквентное (противоправное) поведение'] / max_scores['Делинквентное (противоправное) поведение']) * 10,
        'norms': (scores['Демонстративное игнорирование социальных норм'] / max_scores['Демонстративное игнорирование социальных норм']) * 10,
        'autoaggression': (scores['Аутоагрессия и депрессивные тенденции'] / max_scores['Аутоагрессия и депрессивные тенденции']) * 10,
        'manipulation': (scores['Манипулятивное и аморальное поведение'] / max_scores['Манипулятивное и аморальное поведение']) * 10
    }

    # Расчет нечеткого вывода
    sim = setup_fuzzy_system()
    sim.input['aggression'] = normalized_scores['aggression']
    sim.input['delinquency'] = normalized_scores['delinquency']
    sim.input['norms'] = normalized_scores['norms']
    sim.input['autoaggression'] = normalized_scores['autoaggression']
    sim.input['manipulation'] = normalized_scores['manipulation']

    sim.compute()

    # Создаем график результатов
    fig, ax = plt.subplots(figsize=(10, 6))

    # Бар-чарт результатов
    labels = list(normalized_scores.keys())
    values = [normalized_scores[k] for k in labels]
    ax.bar(labels, values, color=['#e74c3c', '#2980b9', '#2ecc71', '#f1c40f', '#9b59b6'])
    ax.set_ylim(0, 10)
    ax.set_ylabel('Уровень (0-10)')
    ax.set_title('Нормализованные показатели')

    # Конвертируем в base64
    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    result_plot = base64.b64encode(img.getvalue()).decode()
    plt.close()

    return {
        'value': sim.output['deviance'],
        'normalized_scores': normalized_scores,
        'membership_plot': create_membership_plots(),
        'result_plot': result_plot
    }