import matplotlib.pyplot as plt
plt.plot([0,1,2],[1,2,3], label='Observations')
plt.title('Observations')
plt.legend()
plt.savefig('output.png')
