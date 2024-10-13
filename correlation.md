# Define model names and their MSE scores
model_names = ['Linear Regression (LR)', 'ELR-ML']
mse_scores = [test_mse_lr, test_mse_elr_ml]

# Plot bar graph
plt.figure(figsize=(8, 6))
plt.bar(model_names, mse_scores, color=['blue', 'green'])
plt.title('Comparison of Model Performance')
plt.xlabel('Model')
plt.ylabel('Mean Squared Error (MSE)')
plt.xticks(rotation=45)
plt.ylim(0, max(mse_scores) * 1.1)

# Add text labels on top of bars
for i in range(len(model_names)):
    plt.text(i, mse_scores[i] * 1.01, f'{mse_scores[i]:.5f}', ha='center')

plt.tight_layout()
plt.show()
