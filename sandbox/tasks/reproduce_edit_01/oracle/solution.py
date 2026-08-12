import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

data = {
    'Year': [2008, 2008, 2009, 2009, 2010, 2010, 2011, 2011, 2012, 2012, 2013, 2013, 2014, 2014],
    'Coverage': [
        29.8, 11.5, 32.5, 11.5, 31.0, 11.5, 35.0, 12.0,
        34.5, 14.0, 36.5, 14.5, 38.5, 15.0
    ],
    'Source': [
        'Private Bureau', 'Public Registry', 'Private Bureau', 'Public Registry',
        'Private Bureau', 'Public Registry', 'Private Bureau', 'Public Registry',
        'Private Bureau', 'Public Registry', 'Private Bureau', 'Public Registry',
        'Private Bureau', 'Public Registry'
    ]
}

df = pd.DataFrame(data)

plt.figure(figsize=(10, 6))
sns.kdeplot(
    data=df,
    x='Coverage',
    hue='Source',
    fill=True,
    common_norm=False,
    palette={'Private Bureau': '#3498db', 'Public Registry': '#e74c3c'},
    alpha=0.6,
    linewidth=2
)

plt.title('Distribution of Credit Bureau Coverage in Bolivia (2008-2014)', pad=20, fontsize=12)
plt.xlabel('Percentage of Adult Population Covered (%)', fontsize=10)
plt.ylabel('Density', fontsize=10)
plt.xlim(0, 50)
plt.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('out.png', dpi=150, bbox_inches='tight')