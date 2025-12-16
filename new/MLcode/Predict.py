import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import label_binarize, StandardScaler
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
from sklearn.multiclass import OneVsRestClassifier
from imblearn.pipeline import Pipeline as ImbPipeline

# 1. Read Data
data = pd.read_csv("new/ML code/data.csv")
X = data.iloc[:, 0:13].values
Y = data.iloc[:, 14].values  
X_new = np.delete(X, [4, 5], axis=1)
X_n = list(data)[0:13]
X_m = [name for i, name in enumerate(X_n) if i not in [4, 5]]

# 2. Splitting test and training datasets
X_train, X_test, y_train, y_test = train_test_split(
    X_new, Y,
    test_size=0.2,
    random_state=42,
    stratify=Y
)
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

def optimize_model(pipeline, params, X, y):
    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=params,
        scoring='roc_auc_ovr',  
        cv=StratifiedKFold(5, shuffle=True, random_state=42),
        n_jobs=-1,
        error_score='raise'
    )
    grid.fit(X, y)
    return grid.best_estimator_, grid.best_params_

# 3. Optimize SVC model
svc_pipe = ImbPipeline([
    ('sampler', SMOTE(random_state=42)),
    ('scaler', StandardScaler()),
    ('model', SVC(probability=True, decision_function_shape='ovr', class_weight='balanced'))
])
svc_params = {
    'model__C': [0.1, 1, 10],
    'model__kernel': ['linear', 'rbf'],
    'model__gamma': ['scale', 0.1, 1]
}
best_svc, svc_best_params = optimize_model(svc_pipe, svc_params, X_train, y_train)
print("\n=== SVC best ===")
print(svc_best_params)

# 4. Optimize RF model
rf_pipe = ImbPipeline([
    ('sampler', SMOTE(random_state=42)),
    ('model', RandomForestClassifier(class_weight='balanced'))
])
rf_params = {
    'model__n_estimators': [200, 400],
    'model__max_depth': [3, 5, None],
    'model__max_features': ['sqrt', 0.8]
}
best_rf, rf_best_params = optimize_model(rf_pipe, rf_params, X_train, y_train)

# 5. Optimize XGB model
xgb_pipe = ImbPipeline([
    ('sampler', SMOTE(random_state=42)),
    ('model', XGBClassifier(
        objective='multi:softprob',
        eval_metric='mlogloss'))
])
xgb_params = {
    'model__learning_rate': [0.05, 0.1],
    'model__max_depth': [3, 5],
    'model__subsample': [0.8, 1.0]
}
best_xgb, xgb_best_params = optimize_model(xgb_pipe, xgb_params, X_train, y_train)

# 6. Optimize LR model
lr_pipe = ImbPipeline([
    ('sampler', SMOTE(random_state=42)),
    ('scaler', StandardScaler()),
    ('model', OneVsRestClassifier(LogisticRegression(max_iter=10000))) 
])

lr_params = {
    'model__estimator__C': [0.1, 1, 10],       
    'model__estimator__solver': ['lbfgs', 'saga']
}
best_lr, lr_best_params = optimize_model(lr_pipe, lr_params, X_train, y_train)

# 7. Build Ensemble Model
weighted_voting = VotingClassifier(
    estimators=[
        ('svc', best_svc),
        ('rf', best_rf),
        ('xgb', best_xgb),
        ('lr', best_lr)
    ],
    voting='soft'
)
weighted_voting.fit(X_train_res, y_train_res)

# 8. Get prediction probabilities
scaler = StandardScaler()
X_test_scaled = scaler.fit_transform(X_test)
y_score = weighted_voting.predict_proba(X_test_scaled)

y_train_bin = label_binarize(y_train_res, classes=np.unique(y_train_res))
y_test_bin = label_binarize(y_test, classes=np.unique(y_train_res))

n_classes = y_test_bin.shape[1]
print(y_test.shape)
print(y_score.shape)
print(y_test_bin.shape)
weighted_voting.fit(X_new,Y)
explainer = shap.KernelExplainer(weighted_voting.predict, X_new)
shap_values = explainer.shap_values(X_new)

columns = [f'feature{i+1}' for i in range(X_new.shape[1])]
X_df = pd.DataFrame(X_new,columns=columns)
print(shap_values.shape) 
print(X_df.values.shape)

shap_df = pd.DataFrame(shap_values, columns=X_m)
shap_df.to_csv('shap_values.csv', index=False)
print("SHAP values have been saved to 'shap_values.csv'")

# 9. Generate Force Plots for MOL data
explainer = shap.KernelExplainer(
    weighted_voting.predict_proba, 
    X_new, 
    nsamples=1000  
)
mol_data = pd.read_csv("/data/users/lsy/PVK/test.csv")  
X_mol_raw = mol_data.iloc[:, 0:13].values  
X_mol = np.delete(X_mol_raw, [4, 5], axis=1) 
scaler = StandardScaler().fit(X_train_res)
X_mol_scaled = scaler.transform(X_mol)
shap_values_mol = explainer.shap_values(X_mol_scaled)

for i in range(len(X_mol_scaled)):
    sample = X_mol_scaled[i].reshape(1, -1)
    pred_class = weighted_voting.predict(sample)[0]
    
    if isinstance(shap_values_mol, list):
        shap_value = shap_values_mol[pred_class][i]
        base_value = explainer.expected_value[pred_class]
    else:
        shap_value = shap_values_mol[i]
        base_value = explainer.expected_value
    
    force_plot = shap.force_plot(
        base_value=base_value,
        shap_values=shap_value,
        features=X_mol[i],
        feature_names=X_m,
        matplotlib=False
    )
    shap.save_html(f"MOL_force_plot_sample_{i}.html", force_plot)
    
    plt.figure()
    force_plot_static = shap.force_plot(
        base_value=base_value,
        shap_values=shap_value,
        features=X_mol[i],
        feature_names=X_m,
        matplotlib=True
    )
    plt.tight_layout()
    plt.savefig(f"MOL_force_plot_sample_{i}.png", dpi=300, bbox_inches='tight')
    plt.close()

print("All MOL force plots saved.")
print(explainer.expected_value.shape)  

mol_proba = weighted_voting.predict_proba(X_mol_scaled)
print("Unique classes in Y:", np.unique(Y))

target_class = 2  
mol_proba_class2 = mol_proba[:, target_class]

# 10. Save MOL probabilities and generate annotated force plots
mol_results = pd.DataFrame({
    "SampleID": mol_data.index,  
    "Predicted_Label": weighted_voting.predict(X_mol_scaled),
    "Class2_Probability": mol_proba_class2
})
mol_results.to_csv("mol_class2_probabilities.csv", index=False)

for i in range(len(X_mol_scaled)):
    sample = X_mol_scaled[i].reshape(1, -1)
    
    proba = weighted_voting.predict_proba(sample)[0]  
    class2_prob = proba[target_class]
    
    force_plot = shap.force_plot(
        base_value=explainer.expected_value[pred_class],
        shap_values=shap_value,
        features=X_mol_scaled[i],
        feature_names=X_m,
        matplotlib=False,
        show=False
    )
    
    shap.save_html(f"MOL_force_plot_sample_{i}_class2_{class2_prob:.2f}.html", force_plot)

    plt.figure()
    shap.force_plot(
        base_value=explainer.expected_value[pred_class],
        shap_values=shap_value,
        features=X_mol_scaled[i],
        feature_names=X_m,
        matplotlib=True
    )
    plt.title(f"Class 2 Probability: {class2_prob:.2f}", fontsize=12)
    plt.tight_layout()
    plt.savefig(f"MOL_force_plot_sample_{i}_class2_{class2_prob:.2f}.png", dpi=300)
    plt.close()

# 11. Print MOL prediction probabilities
y_proba = weighted_voting.predict_proba(X_mol_scaled)  

for i in range(len(y_proba)):
    print(f" {i} : {y_proba[i]}")