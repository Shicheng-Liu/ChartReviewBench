import matplotlib.pyplot as plt

data = {
    'Sector': ['Fast Fashion', 'Luxury Apparel', 'Automotive ICE', 'Automotive EV',
               'Traditional Build', 'Green Construction', 'Fossil Energy', 'Renewable Energy'],
    'Economic Share': [12.3, 20.8, 28.5, 31.2, 8.6, 12.5, 18.4, 27.7],
    'Growth Category': ['Stagnant', 'Growing', 'Declining', 'Rapid Growth',
                        'Stagnant', 'Rapid Growth', 'Declining', 'Rapid Growth']
}

colors = ['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3',
          '#a6d854', '#ffd92f', '#e5c494', '#b3b3b3']

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

ax1.pie(data['Economic Share'], labels=data['Sector'], autopct='%1.1f%%',
        startangle=90, colors=colors, wedgeprops={'linewidth': 0.5, 'edgecolor': 'white'})
ax1.set_title('Economic Share by Sector (2024)', fontsize=14, pad=20)

growth_dist = {g: sum(s for s, cat in zip(data['Economic Share'], data['Growth Category']) if cat == g)
               for g in set(data['Growth Category'])}
growth_labels = list(growth_dist.keys())
growth_values = list(growth_dist.values())
growth_colors = ['#ffd92f', '#e78ac3', '#66c2a5', '#fc8d62']

ax2.pie(growth_values, labels=growth_labels, autopct='%1.1f%%',
        startangle=90, colors=growth_colors, wedgeprops={'linewidth': 0.5, 'edgecolor': 'white'})
ax2.set_title('Economic Share by Growth Category (2024)', fontsize=14, pad=20)

plt.tight_layout()
plt.savefig('out.png', dpi=300, bbox_inches='tight')