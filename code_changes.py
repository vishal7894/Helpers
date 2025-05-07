# --------------------------- 1  cell ----------------------------------------- 

features = ["University_enc", "Gender_enc", "CustomerAge@PostDate", "months_since_last_deposit", 
            "last_3m_avg_balance", "balance_monthly_index", "balance_quarterly_index", 
            "last_3m_avg_d_spend", "d_spend_monthly_index", "d_spend_quarterly_index", 
            "last_3m_avg_p_spend", "p_spend_monthly_index", "p_spend_quarterly_index", 
            "logins_monthly_index", "logins_quarterly_index" ]

le_uni = LabelEncoder()
le_gender = LabelEncoder()

Train_agg_df['University_enc'] = le_uni.fit_transform(Train_agg_df['University'])
Train_agg_df['Gender_enc'] = le_gender.fit_transform(Train_agg_df['Gender'])

train_agg_df['University_enc'] = le_uni.fit_transform(train_agg_df['University'])
train_agg_df['Gender_enc'] = le_gender.fit_transform(train_agg_df['Gender'])

test_agg_df['University_enc'] = le_uni.fit_transform(test_agg_df['University'])
test_agg_df['Gender_enc'] = le_gender.fit_transform(test_agg_df['Gender'])
# --------------------------- 1  cell -----------------------------------------


# --------------------------- 2  cell -----------------------------------------
def train_evaluate_model(X_train, y_train, X_test, y_test, cut_off = 0.5):
    clf = lgb.LGBMClassifier(random_state=42, n_estimators=100)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= cutoff).astype(int)
    
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("ROC AUC:", roc_auc_score(y_test, y_proba))
    lgb.plot_importance(clf)
    plt.show()

    # Confusion matrix
    cm = confusion_matrix(y_pred, y_test)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt="d",cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()

    # Save the trained model
    joblib.dump(clf, 'trained_model.pkl')
    
    # Create prediction dataframe
    prediction_df = test_df[['ReferenceDate', 'Customernumber']].copy()
    prediction_df['prediction'] = y_pred
    prediction_df['probability_predicted'] = y_proba

    return prediction_df


def predict_future(X_train, model_path, cut_off = 0.5):

    # Load the saved model
    clf = joblib.load(model_path)
    y_pred = clf.predict(X_train)
    # Predict probabilities
    y_proba = clf.predict_proba(X_train)[:, 1]
    # Apply custom cutoff
    cutoff = 0.7  # or whatever you used during training
    y_pred_adjusted = (y_proba >= cutoff).astype(int)

    prediction_df = X_train[['ReferenceDate', 'Customernumber']].copy()
    prediction_df['prediction'] = y_pred_adjusted
    prediction_df['probability'] = y_proba

    return prediction_df

# --------------------------- 2  cell -----------------------------------------

# --------------------------- 3  cell -----------------------------------------
x_train, y_train = train_agg_df[features], train_agg_df['target']
x_test, y_test = test_agg_df[features], test_agg_df['target']
X_train, Y_train = Train_agg_df[features], Train_agg_df['target']

prediction_df = train_evaluate_model(x_train, y_train, x_test, y_test, cut_off = 0.5)

final_df = predict_future(X_train, model_path, cut_off = 0.5)

# --------------------------- 3  cell -----------------------------------------
