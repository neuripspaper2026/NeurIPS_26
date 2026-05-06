import matplotlib.pyplot as plt
import seaborn as sns
import os

# 设置画图风格
sns.set_style("whitegrid")
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10
})

# 配置
FIGURE_DIR = "../../analysis_summaries/figures"
os.makedirs(FIGURE_DIR, exist_ok=True)

TABLE_DIR = "../../analysis_summaries/tables"

EX_VERSIONS = ['EX1', 'EX2']
DATASET_SIZES = ['mini', 'small', 'medium', 'large', 'extra-large']
MODELS = ['claude', 'gpt5.1', 'qwen']

# 颜色方案
MODEL_COLORS = {
    'claude': '#1f77b4',    # 蓝色
    'gpt5.1': '#ff7f0e',    # 橙色
    'qwen': '#2ca02c'       # 绿色
}

MODEL_LABELS = {
    'claude': 'Claude',
    'gpt5.1': 'GPT-5.1',
    'qwen': 'Qwen'
}

print("✓ Plot configuration completed")
print(f"  Figure directory: {FIGURE_DIR}")
print(f"  Table directory: {TABLE_DIR}")

def plot_metric_vs_p_thr(metric_name='Fast_at_k_avg', 
                          ylabel='fast@3',
                          title_prefix='fast@3',
                          filename='figure_fast_at_k_all_datasets.pdf'):
    """
    画 metric vs p_thr 的图，2行5列布局
    
    Args:
        metric_name: 要画的指标列名 ('Fast_at_k_avg' 或 'Speedup_at_k_avg')
        ylabel: Y轴标签
        title_prefix: 图表标题前缀
        filename: 保存的文件名
    """
    
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    #fig.suptitle(f'{title_prefix} 随 p_thr 的变化 (k=3)', fontsize=16, y=0.995)
    
    # 遍历所有 EX × dataset_size 组合
    for row_idx, ex in enumerate(EX_VERSIONS):
        for col_idx, ds in enumerate(DATASET_SIZES):
            ax = axes[row_idx, col_idx]
            
            # 读取对应的 Table E 文件
            table_e_file = f"{TABLE_DIR}/table_e_paper_level_k3_{ex}_{ds}.csv"
            
            if not os.path.exists(table_e_file):
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center')
                ax.set_title(f'{ds}\n({ex})')
                continue
            
            df_e = pd.read_csv(table_e_file)
            
            # 对于 EX2，只保留 threads=16 的数据
            if ex == 'EX2':
                df_e = df_e[df_e['threads'] == 16.0]
            
            # 为每个 model 画一条线
            for model in MODELS:
                df_model = df_e[df_e['model'] == model].sort_values('p_thr')
                
                if len(df_model) == 0:
                    continue
                
                ax.plot(df_model['p_thr'], 
                       df_model[metric_name],
                       marker='o',
                       linewidth=2,
                       markersize=4,
                       label=MODEL_LABELS[model],
                       color=MODEL_COLORS[model],
                       alpha=0.8)
            
            # 设置子图标题和标签
            ax.set_title(f'{ds}\n({ex})', fontsize=11)
            ax.set_xlabel('p_thr', fontsize=10)
            ax.set_ylabel(ylabel, fontsize=10)
            ax.grid(True, alpha=0.3)
            
            # 只在第一个子图显示图例
            if row_idx == 0 and col_idx == 0:
                ax.legend(loc='best', framealpha=0.9)
            
            # 设置 Y 轴范围
            if metric_name == 'Fast_at_k_avg':
                ax.set_ylim(-0.05, 1.05)
            else:
                ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    
    # 保存图片
    output_path = f"{FIGURE_DIR}/{filename}"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Figure saved: {output_path}")
    
    plt.show()


print("✓ Plot function defined")


print("="*80)
print("Plot: Fast@3 vs p_thr (RQ1)")
print("="*80)

plot_metric_vs_p_thr(
    metric_name='Fast_at_k_avg',
    ylabel='fast@3',
    title_prefix='fast@3',
    filename='figure_fast_at_k_all_datasets.pdf'
)