import matplotlib.pyplot as plt
import seaborn as sns


def plot_models_data(dp, filename, metrics_names, behav=False, store_id='0'):
    """
    修改后的绘图函数：每个指标一个柱状图，每个柱状图包含多个模型
    """
    model_order = ['local_llama1B', 'local_llama8B', 'centaur', 'mixratio']  # 指定顺序
    tasks = metrics_names  # 指标即任务
    n_tasks = len(tasks)

    # 计算子图布局，确保图表合理显示
    n_cols = min(3, n_tasks)  # 每行最多3个图
    n_rows = (n_tasks + n_cols - 1) // n_cols  # 计算行数

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows), sharex=False, sharey=True)
    axs = axs.flatten()  # 展平子图数组以便遍历

    colors = ['tab:green', 'tab:orange', 'tab:blue', 'tab:red', 'tab:brown', 'tab:purple']

    for i, task in enumerate(tasks):
        ax = axs[i]
        dp_task = dp[dp['Task'] == task]

        # 在barplot中添加order参数
        sns.barplot(
            data=dp_task,
            x='Model',
            y='Value',
            ax=ax,
            alpha=0.6,
            hue='Model',
            palette=colors,
            dodge=False,
            order=model_order  # 强制指定柱状图顺序
        )

        # 更新横坐标标签
        new_labels = ['LlaMA_1B', 'LlaMA_8B', 'centaur', 'mixratio']
        ax.set_xticklabels(new_labels, rotation=45, fontsize=9)  # 使用新的顺序
        # 添加误差条
        for j, model in enumerate(model_order):  # 按新顺序遍历
            model_data = dp_task[dp_task['Model'] == model]
            if not model_data.empty:
                ci = model_data['CI'].values[0]
                value = model_data['Value'].values[0]
                ax.errorbar(j, value, yerr=ci / 2, color='black', capsize=2, fmt='none')

        # 设置图表标题和标签
        ax.set_title(task, fontsize=10)
        ax.axhline(y=1, color='black', linestyle='dotted')
        ax.axhline(y=0, color='black', linestyle='dotted')

        if behav:
            ax.set_ylim(-1, 5)
        else:
            ax.set_ylim(-1, 3)

        # 调整标签显示
        ax.set_ylabel('Normalized Value (Random=0, Human=1)')
        if i >= n_cols * (n_rows - 1):  # 底部子图显示x轴标签
            ax.set_xlabel('Models')
        else:
            ax.set_xlabel('')

        # 移除图例，只在第一个子图显示
        if i == 0:
            ax.legend(title='Models', bbox_to_anchor=(1.05, 1), loc='upper left')
        else:
            ax.legend().remove()

    # 隐藏多余的子图
    for i in range(n_tasks, len(axs)):
        axs[i].axis('off')

    plt.tight_layout()
    plt.savefig(f'./plots/models/{store_id}{filename}')
