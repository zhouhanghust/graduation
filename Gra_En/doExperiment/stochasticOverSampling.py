from imblearn.over_sampling import RandomOverSampler


X =  [[2,2,2,2,2],[4,4,4,4,4],[6,6,6,6,6],[1,1,1,1,1],[3,3,3,3,3],[8,8,8,8,8]]
y = [0,0,0,1,1,0]


ros = RandomOverSampler(random_state=0)
X_resampled, y_resampled = ros.fit_sample(X, y)

print(X_resampled)
print("--------------------------------------")
print(y_resampled)






