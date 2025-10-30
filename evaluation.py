import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import NullLocator
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score,
    root_mean_squared_error,
)


def calculate_importance_gains(model, columns) -> pd.DataFrame:
    """Вычисляет важность признаков (information gain) в процентном выражении.

    Вычисляет относительную важность признаков из обученной модели, преобразует их
    в проценты от общей важности и возвращает отсортированный DataFrame.

    Parameters
    ----------
    model : decision tree model like
        Модель, из которой извлекаются важности признаков. Должна иметь атрибут feature_importances_
    columns : list-like
        Названия признаков, соответствующие индексам feature_importances.

    Returns
    -------
    pd.DataFrame
        _description_
    """
    feature_importances_percent = pd.DataFrame(
        data=(
            model.feature_importances_ / model.feature_importances_.sum() * 100
        ).round(3),
        index=columns,
        columns=["information gain (%)"],
    ).sort_values(by="information gain (%)", ascending=False)
    return feature_importances_percent


def plot_feature_importances(
    feature_importances_percent: pd.DataFrame, font_size: int = 18
):
    """Строит столбчатую диаграмму важности признаков.

    Визуализирует важность признаков в виде горизонтальной столбчатой диаграммы,
    где признаки отсортированы по степени важности (information gain).

    Parameters
    ----------
    feature_importances_percent : pd.DataFrame
        DataFrame с важностью признаков в процентном выражении.
        Должен содержать колонку 'information gain (%)' и индекс с названиями признаков.
    font_size : int
        Размер шрифта для заголовков и подписей осей, по умолчанию 18.
    """
    plt.figure(figsize=(19, 6))
    ax = sns.barplot(
        feature_importances_percent,
        x="information gain (%)",
        y=feature_importances_percent.index,
    )
    ax.xaxis.set_major_locator(NullLocator())
    plt.title("Значимость факторов", fontsize=font_size)
    plt.xlabel("Information Gain", fontsize=font_size)
    plt.ylabel("Фактор", fontsize=font_size)
    plt.yticks(fontsize=font_size)
    plt.show()


def evaluate_regression(y_true, y_pred, feautures_count: int) -> None:
    """Выводит основные метрики качества регрессионной модели.

    Вычисляет и отображает метрики:
    - RMSE (Root Mean Squared Error) - среднеквадратичная ошибка;
    - MAE (Mean Absolute Error) - средняя абсолютная ошибка;
    - MAPE (Mean Absolute Percentage Error) - средняя абсолютная процентная ошибка;
    - Adjusted R2 (скорректированный коэффициент детерминации).

    Parameters
    ----------
    y_true : np.array
        Истинные значения целевой переменной.
    y_pred : np.array
        Предсказанные значения целевой переменной.
    feautures_count : int
        Количество признаков, используемых в модели (для расчета Adjusted R2).
    """
    RMSE = root_mean_squared_error(y_true, y_pred)
    print(f"RMSE равняется {RMSE:.3f}")
    MAE = mean_absolute_error(y_true, y_pred)
    print(f"MAE равняется {MAE:.3f}")
    MAPE = mean_absolute_percentage_error(y_true, y_pred) * 100
    print(f"MAPE равняется {MAPE:.3f} %")
    N: int = len(y_pred)
    R2 = r2_score(y_true, y_pred)
    Adj_R2 = 1 - (1 - R2) * (N - 1) / (N - feautures_count - 1)
    print(f"Adjusted R2 равняется {Adj_R2:.3f}")


def plot_regression(y_true, y_pred, font_size: int = 18) -> None:
    """Визуализирует качество регрессионной модели с помощью диаграммы рассеяния.

    Строит график зависимости предсказанных значений от фактических, добавляет
    линию регрессии и отображает коэффициент корреляции между предсказаниями и целевыми значениями.

    Parameters
    ----------
    y_true : np.array
        Фактические значения целевой переменной.
    y_pred : np.array
        Предсказанные значения целевой переменной.
    font_size : int
        Размер шрифта для заголовков и подписей осей, по умолчанию 18.
    """
    plt.figure(figsize=(19, 6))
    coef_corr = np.corrcoef(y_true, y_pred)[0, 1]
    plt.title(f"Коэффициент корреляции равен {coef_corr:.3f}", fontsize=font_size)
    sns.regplot(x=y_true, y=y_pred, ci=None, line_kws=dict(color="r", linestyle="--"))
    plt.xlabel("Фактические значения ($)", fontsize=font_size)
    plt.ylabel("Прогнозные значения ($)", fontsize=font_size)
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)
    plt.grid(axis="y")
    plt.show()


def residuals_histogram(
    y_true,
    y_pred,
    font_size: int = 18,
) -> None:
    """Строит гистограмму распределения остатков регрессионной модели.

    Визуализирует распределение ошибок (разностей между фактическими и предсказанными значениями)
    с помощью гистограммы и кривой плотности. Отображает стандартное отклонение целевой переменной.

    Parameters
    ----------
    y_true : np.array
        Фактические значения целевой переменной.
    y_pred : np.array
        Предсказанные значения целевой переменной.
    font_size : int, optional
        Размер шрифта для заголовков и подписей осей, по умолчанию 18.
    """
    errors = y_true - y_pred
    plt.figure(figsize=(19, 6))
    ax = sns.histplot(x=errors, kde=True, bins=100)
    ax.lines[0].set_color("red")
    plt.title(
        rf"Гистрограмма распределения остатков, $\sigma$ = {y_true.std():.3f}",
        fontsize=font_size,
    )
    plt.xticks(fontsize=font_size)
    plt.xlabel("Отклонения ($)", fontsize=font_size)
    plt.yticks(fontsize=font_size)
    plt.ylabel("Количество", fontsize=font_size)
    plt.show()
