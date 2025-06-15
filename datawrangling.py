# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis

# %%
file_path = "/Users/tomamely/Desktop/Ironhack 2025/Project_A.Btesting/df_final_demo.csv"
df_final = pd.read_csv(file_path)
print(f"size: {df_final.shape}") 
df_final.head()

# %%
df_final['client_id'].nunique() 

# %%
print(df_final.isna().sum())

# %%
df_final_clean = df_final.dropna()
print(f"Number of rows after drop: {df_final_clean.shape[0]}")

# %%
print(df_final_clean.dtypes)

# %%
df_final_clean['gendr'] = df_final_clean['gendr'].replace('X', 'U')

# %%
print(df_final_clean['gendr'].value_counts())

# %%
active_clients = df_final_clean[df_final_clean['logons_6_mnth'] >= 5]
print("active clients avarege age :", active_clients['clnt_age'].mean())
print("Mean years being active:", active_clients['clnt_tenure_yr'].mean())
print("Genders:\n", active_clients['gendr'].value_counts(normalize=True))

# %%
file_path = "/Users/tomamely/Desktop/Ironhack 2025/Project_A.Btesting/df_final_experiment_clients.csv"
df_clients = pd.read_csv(file_path)
df_clients.head()

# %%
print(df_clients.isna().sum())

# %%
print(df_clients.shape[0])

# %%
print(df_clients['Variation'].value_counts())

# %%
merged = pd.merge(df_clients, df_final_clean[['client_id', 'gendr']], on='client_id', how='left')

# %%
grouped = merged.groupby(['gendr', 'Variation']).size().reset_index(name='count')

# %%
print(grouped)

# %%

counts = merged.groupby(['Variation', 'gendr']).size().reset_index(name='count')

pivot = counts.pivot(index='gendr', columns='Variation', values='count').fillna(0).astype(int).reset_index()

print(pivot)

# %%
print("df1:", df_final_clean['client_id'].nunique())
print("df2:", df_clients['client_id'].nunique())
print("merged:", merged['client_id'].nunique())

# %%
file_path = "/Users/tomamely/Desktop/Ironhack 2025/Project_A.Btesting/df_final_web_data_pt_1.csv"
df1 = pd.read_csv(file_path)
df1.head()

# %%
file_path = "/Users/tomamely/Desktop/Ironhack 2025/Project_A.Btesting/df_final_web_data_pt_2.csv"
df2 = pd.read_csv(file_path)
df2.head()

# %%
df_combined = pd.concat([df1, df2], ignore_index=True)

# %%
df_combined.head()

# %%
# datetime
df_combined['date_time'] = pd.to_datetime(df_combined['date_time'])

# nulls
print(df_combined.info())
print("\nnulls:")
print(df_combined.isna().sum())

# sort before finding duplicates
df_combined = df_combined.sort_values(by=['client_id', 'visit_id', 'date_time'])

# finding duplicates
duplicates_all = df_combined[df_combined.duplicated(subset=['client_id', 'visit_id', 'process_step', 'date_time'], keep=False)]
print(f"All duplicated rows: {duplicates_all.shape[0]}")
print(duplicates_all.sort_values(by=['client_id', 'visit_id', 'date_time']).head(10))

# deleting duplicates
df_combined = df_combined.drop_duplicates(subset=['client_id', 'visit_id', 'process_step', 'date_time'])
print(f"After delete: {df_combined.shape}")




# %%
# date is correctly formatted
df_combined['date_time'] = pd.to_datetime(df_combined['date_time'])




# %%
print("Before:", df_combined.shape[0])
df = df_combined.drop_duplicates(subset=['client_id', 'visitor_id', 'visit_id', 'process_step', 'date_time'], keep='first')
print("After:", df.shape[0])

# %%
df['client_id'].nunique() 

# %%
# with df_clients
df_combined = df_combined.merge(df_clients, on='client_id', how='left')
print(df_combined['process_step'].value_counts())

# %%
df_combined = df_combined.sort_values(by=['visit_id', 'date_time'])
df_combined.head()

# %%

df_combined['next_time'] = df_combined.groupby(['visit_id'])['date_time'].shift(-1)
df_combined['duration'] = (df_combined['next_time'] - df_combined['date_time']).dt.total_seconds()


avg_duration = df_combined.groupby(['Variation', 'process_step'])['duration'].mean().unstack(0)


# %%
# Check if each client has reached the 'confirm' step
client_confirm = df_combined.groupby("client_id")["process_step"].apply(lambda steps: "confirm" in steps.values)
print(client_confirm.head())

# %%
counts = client_confirm.value_counts()
num_reached = counts.get(True, 0)
num_not_reached = counts.get(False, 0)
print(f"Clients who reached 'confirm': {num_reached}")
print(f"Clients who did NOT reach 'confirm': {num_not_reached}")

# %%

client_confirm = df_combined.groupby('visit_id')['process_step'].apply(lambda x: 'confirm' in x.values)


counts = client_confirm.value_counts()
num_reached = counts.get(True, 0)
num_not_reached = counts.get(False, 0)

completion_rate_clients = num_reached / (num_reached + num_not_reached)
print(f"Completion Rate (by client): {completion_rate_clients:.2%}")


# %%
# Группируем по visit_id и Variation, проверяем наличие 'confirm'
visit_confirm = df_combined.groupby(['visit_id', 'Variation'])['process_step'].apply(lambda x: 'confirm' in x.values).reset_index(name='reached_confirm')

# Считаем completion rate по группам Variation
completion_rate_by_variation = (
    visit_confirm.groupby('Variation')['reached_confirm']
    .mean()
    .round(4) * 100
)

# Выводим результат
print("Completion Rate by Variation (%):")
print(completion_rate_by_variation)




# %%
#H0 = comp rate is the same for Test and Control groups p_test=p_control
#H1 = comp rate is higher  for the Test group than for the Control group p_test>p_control
#Q-ty
#Test       26968
#Control    23532
#Variation
#Control    49.85%
#Test       58.52%


# %%
from statsmodels.stats.proportion import proportions_ztest
n_control = 23532
n_test = 26968
success_control = round(49.85/100 * n_control)
success_test = round(58.52/100 * n_test)
alpha = 0.05
stat, p_value = proportions_ztest([success_test, success_control], [n_test, n_control], alternative='larger')
print(f"Z-statistic: {stat:.4f}")
print(f"P-value: {p_value:.4e}")



# %%
if p_value < alpha:
    print("Reject H0 → The new design is statistically significantly better")
else:
    print("Fail to reject H0 → The difference is not statistically significant")

# %%

# H0: p_test - p_control <= 0.05
# H1: p_test - p_control > 0.05

n_control = 23532
n_test = 26968

p_control = 49.85 / 100
p_test = 58.52 / 100

success_control = round(p_control * n_control)
success_test = round(p_test * n_test)


min_required_diff = 0.05

alpha = 0.05

stat, p_value = proportions_ztest([success_test, success_control],[n_test, n_control],value=min_required_diff,alternative='larger')

print(f"Z-statistic: {stat:.4f}")
print(f"P-value: {p_value:.4e}")


# %%
if p_value < alpha:
    print("Reject H0 → The increase in completion rate is statistically significantly greater than 5% of the control rate.")
    print("Conclusion: The new design is economically justifiable based on the observed improvement.")
else:
    print("Fail to reject H0 → The increase in completion rate is not significantly greater than 5% of the control rate.")
    print("Conclusion: The new design may not be cost-effective based on the current data.")


# %%
# Sort the dataframe by 'visit_id' and 'date_time' to ensure correct order of steps within each visit
df_combined = df_combined.sort_values(by=['visit_id', 'date_time'])

# Create a new column 'next_date_time' which contains the timestamp of the next step within the same visit
df_combined['next_date_time'] = df_combined.groupby('visit_id')['date_time'].shift(-1)

# Calculate the duration spent on the current step in seconds by subtracting current time from next step's time
df_combined['step_duration'] = (df_combined['next_date_time'] - df_combined['date_time']).dt.total_seconds()

# Remove rows where 'step_duration' is NaN (these are the last steps in each visit with no next step)
step_durations = df_combined.dropna(subset=['step_duration'])

# Compute the average time spent per step grouped by 'Variation' and 'process_step'
avg_time_per_step_by_group = (
    step_durations.groupby(['Variation', 'process_step'])['step_duration']
    .mean()
    .round(2)
    .unstack(level=0)  # Unstack 'Variation' to columns for easier comparison
)

print("Average Time per Step (seconds) by Variation:")
print(avg_time_per_step_by_group)


# %%
df_sorted = df_combined.sort_values(['visit_id', 'date_time'])
df_sorted['step_duration'] = df_sorted.groupby('visit_id')['date_time'].shift(-1) - df_sorted['date_time']
df_sorted['step_duration'] = df_sorted['step_duration'].dt.total_seconds()

avg_time_per_step = (
    df_sorted.dropna(subset=['step_duration'])
    .groupby(['Variation', 'process_step'])['step_duration']
    .mean()
    .round(2)
    .unstack('Variation')
)

print("Average Time per Step (seconds) by Variation:")
print(avg_time_per_step)

# %%
# Define the desired order of process steps from start to confirm
step_order = ['start', 'step_1', 'step_2', 'step_3', 'confirm']

# Sort the average times by this order
avg_time_per_step_by_group = avg_time_per_step_by_group.reindex(step_order)

print("Average Time per Step (seconds) by Variation (ordered):")
avg_time_per_step_by_group

# %%
# Сортируем по visit_id и времени
df_sorted = df_combined.sort_values(by=['visit_id', 'date_time'])

# Присваиваем порядковые номера шагам
step_order = {step: i for i, step in enumerate(df_sorted['process_step'].unique())}
df_sorted['step_num'] = df_sorted['process_step'].map(step_order)

# Сдвигаем предыдущий шаг внутри визита
df_sorted['prev_step_num'] = df_sorted.groupby('visit_id')['step_num'].shift(1)

# Определяем, был ли возврат назад (ошибка)
df_sorted['is_error'] = df_sorted['step_num'] < df_sorted['prev_step_num']

# Считаем среднюю долю ошибок по группе
error_rate_by_group = (
    df_sorted.groupby('Variation')['is_error']
    .mean()
    .round(4)
)

# Вывод
print("Error Rate by Variation:")
print((error_rate_by_group * 100).astype(str) + '%')


# %%
step_order = {'start': 0, 'step_1': 1, 'step_2': 2, 'step_3': 3, 'confirm': 4}
df_sorted['step_num'] = df_sorted['process_step'].map(step_order)
df_sorted = df_sorted.sort_values(by=['visit_id', 'date_time'])
df_sorted['prev_step_num'] = df_sorted.groupby('visit_id')['step_num'].shift(1)
df_sorted['is_error'] = df_sorted['step_num'] < df_sorted['prev_step_num']

# %%

visits_with_error = df_sorted.groupby('visit_id')['is_error'].any().reset_index()

error_rate = visits_with_error['is_error'].mean().round(4)

print(f"Error Rate (per visit): {error_rate * 100:.2f}%")


# %%
# Проверим: был ли хотя бы один is_error в рамках визита
visits_with_error = df_sorted.groupby(['visit_id', 'Variation'])['is_error'].any().reset_index()

# Группируем по Variation и считаем среднюю долю визитов с ошибкой
error_rate_by_visits = (
    visits_with_error.groupby('Variation')['is_error']
    .mean()
    .mul(100)  # сразу умножаем на 100
    .round(2)  # округляем до 2 знаков
)

# Выводим результат
print("Error Rate by Variation (per visit):")
print(error_rate_by_visits.astype(str) + '%')

# %%
# Step order mapping
step_order = {'start': 0, 'step_1': 1, 'step_2': 2, 'step_3': 3, 'confirm': 4}
df_sorted['step_num'] = df_sorted['process_step'].map(step_order)
# Sort properly
df_sorted = df_sorted.sort_values(by=['visit_id', 'date_time'])
# Calculate step difference per visit
df_sorted['prev_step_num'] = df_sorted.groupby('visit_id')['step_num'].shift(1)
df_sorted['step_diff'] = df_sorted['step_num'] - df_sorted['prev_step_num']
# Mark row-level backward steps
df_sorted['step_error'] = df_sorted['step_diff'] < 0
# Aggregate to one error per visit
visit_errors = df_sorted.groupby('visit_id')['step_error'].any().reset_index()
visit_errors['visit_has_error'] = visit_errors['step_error'].astype(int)
# Compute visit-level KPI
total_visits = visit_errors.shape[0]
visits_with_errors = visit_errors['visit_has_error'].sum()
visit_level_error_rate = visits_with_errors / total_visits if total_visits > 0 else 0
print(f"Total visits: {total_visits}")
print(f"Visits with errors: {visits_with_errors}")
print(f"Visit-level error rate: {visit_level_error_rate:.2%}")

# %%
step_order = {step: i for i, step in enumerate(df_combined['process_step'].unique())}
df_combined['step_index'] = df_combined['process_step'].map(step_order)

# Разница между текущим и предыдущим шагом
df_combined['prev_step_index'] = df_combined.groupby(['client_id', 'visit_id'])['step_index'].shift(1)
df_combined['step_diff'] = df_combined['step_index'] - df_combined['prev_step_index']

# Переход назад — если разница < 0
df_combined['error'] = df_combined['step_diff'] < 0

# Error rate по вариациям
error_rate = df_combined.groupby('Variation')['error'].mean()
print("Error Rate by Variation:\n", error_rate)

# %%
df_tableu = pd.merge(df_combined, df_final_clean, on='client_id', how='left')
df_tableu.head()

# %%
df = df_tableu.copy()

step_order = {'start': 0, 'step_1': 1, 'step_2': 2, 'step_3': 3, 'confirm': 4}
df['step_num'] = df['process_step'].map(step_order)

df = df.sort_values(['visit_id', 'date_time'])

df['prev_step_num'] = df.groupby('visit_id')['step_num'].shift(1)
df['is_error'] = df['step_num'] < df['prev_step_num']

visit_flags = df.groupby(['visit_id', 'Variation']).agg(
    reached_confirm=('process_step', lambda x: 'confirm' in x.values),
    is_error=('is_error', 'any')
).reset_index()

df_final = df.merge(
    visit_flags[['visit_id', 'Variation', 'reached_confirm', 'is_error']],
    on=['visit_id', 'Variation'],
    how='left'
)



# %%
df_final.head()

# %%
df_final = df_final.rename(columns={
    'is_error_y': 'is_error',
    'reached_confirm': 'reached_confirm'
})

# %%
visit_level = df_final.groupby(['visit_id', 'Variation']).agg(
    reached_confirm=('reached_confirm', 'max'),  # или 'any'
    is_error=('is_error', 'max')  # или 'any'
).reset_index()

completion_rate = (visit_level.groupby('Variation')['reached_confirm'].mean() * 100).round(2)
error_rate = (visit_level.groupby('Variation')['is_error'].mean() * 100).round(2)

print("Completion Rate (%):")
print(completion_rate)

print("\nError Rate (%):")
print(error_rate)


# %%
# Заполняем NaN
df_final['Variation'] = df_final['Variation'].fillna('Unknown')
df_final['reached_confirm'] = df_final['reached_confirm'].fillna(False)
df_final['is_error'] = df_final['is_error'].fillna(False)

# Группируем на уровне визита
visit_level = df_final.groupby(['visit_id', 'Variation']).agg(
    reached_confirm_visit = ('reached_confirm', 'max'),  # есть ли confirm в визите
    is_error_visit = ('is_error', 'max')  # была ли ошибка в визите
).reset_index()

# Считаем KPI по Variation
completion_rate = (visit_level.groupby('Variation')['reached_confirm_visit'].mean() * 100).round(2)
error_rate = (visit_level.groupby('Variation')['is_error_visit'].mean() * 100).round(2)

print("Completion Rate (%):")
print(completion_rate)

print("\nError Rate (%):")
print(error_rate)


# %%
print(df_final[['reached_confirm', 'is_error']].dtypes)
print(df_final[['reached_confirm', 'is_error']].isnull().sum())
print(df_final['visit_id'].nunique())
print(df_final.shape[0])

# %%
print(df_final['visit_id'].nunique())

# %%
visit_level.groupby('Variation')['visit_id'].nunique()


# %%
visit_level = df_final.groupby(['visit_id', 'Variation']).agg(
    is_error_flag_num=('is_error', lambda x: int(any(x)))
).reset_index()

visit_level['Visit_Error_Flag'] = visit_level['is_error_flag_num']

# Теперь KPI
error_rate = (visit_level.groupby('Variation')['Visit_Error_Flag'].mean() * 100).round(2)
print(error_rate)

# %%
visit_level.head()

# %%
df_final.to_csv("df_final.csv", index=False, encoding='utf-8')

# %%
from scipy import stats

# Разделим выборки по Variation
age_test = df_tableu[df_tableu['Variation'] == 'Test']['clnt_age'].dropna()
age_control = df_tableu[df_tableu['Variation'] == 'Control']['clnt_age'].dropna()


# T-test на равенство средних
stat, p_value = stats.ttest_ind(age_test, age_control, equal_var=False)

stat, p_value

# %%
sns.violinplot(x='Variation', y='clnt_age', data=df_tableu)

# %%
print("Среднее Test:", age_test.mean())
print("Среднее Control:", age_control.mean())

# %%
from scipy.stats import mannwhitneyu

stat, p_value = mannwhitneyu(age_test, age_control, alternative='two-sided')
print("U-тест: stat =", stat, ", p =", p_value)


# %%
from scipy.stats import ks_2samp

stat, p_value = ks_2samp(age_test, age_control)
print("KS-тест: stat =", stat, ", p =", p_value)


# %%
import numpy as np

def cohens_d(x, y):
    nx, ny = len(x), len(y)
    pooled_std = np.sqrt(((nx - 1)*np.var(x, ddof=1) + (ny - 1)*np.var(y, ddof=1)) / (nx + ny - 2))
    return (np.mean(x) - np.mean(y)) / pooled_std

print("Cohen's d:", cohens_d(age_test, age_control))

# %%
print("Размер Test:", len(age_test))
print("Размер Control:", len(age_control))

# %%
import seaborn as sns
import matplotlib.pyplot as plt

sns.kdeplot(age_test, label='Test', fill=True, common_norm=False)
sns.kdeplot(age_control, label='Control', fill=True, common_norm=False)
plt.title("Age Distribution Density")
plt.xlabel("Age")
plt.ylabel("Density")
plt.legend(title="Group")
plt.show()

# %%
import seaborn as sns
import matplotlib.pyplot as plt

# Построение плотностей распределения возраста по группам
sns.kdeplot(age_control, label='Control', fill=True, color='#7f7f7f')  # серый
sns.kdeplot(age_test, label='Test', fill=True, color='#2ca02c')        # зелёный

# Настройка графика
plt.title("Age Distribution: Control vs Test")
plt.xlabel("Age")
plt.ylabel("Density")
plt.legend(title="Group")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()



