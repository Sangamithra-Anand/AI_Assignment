"""
analyze_rules.py
-------------------------
Generates:
1. Visualizations -> output/visuals/
2. output/insights.txt
3. output/reports/groceries_report.txt
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx


def analyze_rules(rules):
    print("[INFO] Starting rule analysis...")

    # -----------------------------
    # SUPPORT DISTRIBUTION
    # -----------------------------
    sns.histplot(rules["support"], kde=True)
    plt.title("Support Distribution")
    plt.savefig("output/visuals/support_distribution.png")
    plt.clf()

    # -----------------------------
    # CONFIDENCE DISTRIBUTION
    # -----------------------------
    sns.histplot(rules["confidence"], kde=True)
    plt.title("Confidence Distribution")
    plt.savefig("output/visuals/confidence_distribution.png")
    plt.clf()

    # -----------------------------
    # LIFT SCATTER PLOT
    # -----------------------------
    plt.scatter(rules["support"], rules["lift"], alpha=0.6)
    plt.title("Lift vs Support")
    plt.savefig("output/visuals/lift_scatter_plot.png")
    plt.clf()

    # -----------------------------
    # NETWORK GRAPH
    # -----------------------------
    graph = nx.DiGraph()

    for _, row in rules.iterrows():
        left = ', '.join(list(row['antecedents']))
        right = ', '.join(list(row['consequents']))
        graph.add_edge(left, right, weight=row['lift'])

    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(graph, k=1)
    nx.draw(graph, pos, with_labels=True, node_size=2500, font_size=8, font_weight="bold")

    plt.title("Association Rule Network Graph")
    plt.savefig("output/visuals/network_graph.png")
    plt.clf()

    # -----------------------------
    # STRONGEST & MOST CONFIDENT RULE
    # -----------------------------
    strongest_rule = rules.sort_values(by="lift", ascending=False).iloc[0]
    most_confident_rule = rules.sort_values(by="confidence", ascending=False).iloc[0]

    # -----------------------------
    # WRITE insights.txt
    # -----------------------------
    with open("output/insights.txt", "w", encoding="utf-8") as f:
        f.write("ASSOCIATION RULE INSIGHTS\n")
        f.write("--------------------------\n\n")
        f.write(f"Total Rules Generated: {len(rules)}\n\n")

        f.write("Strongest Rule (by Lift):\n")
        f.write(f"  {list(strongest_rule['antecedents'])} -> {list(strongest_rule['consequents'])}\n")
        f.write(f"  Lift: {strongest_rule['lift']:.3f}\n")
        f.write(f"  Confidence: {strongest_rule['confidence']:.3f}\n\n")

        f.write("Most Confident Rule:\n")
        f.write(f"  {list(most_confident_rule['antecedents'])} -> {list(most_confident_rule['consequents'])}\n")
        f.write(f"  Confidence: {most_confident_rule['confidence']:.3f}\n")
        f.write(f"  Support: {most_confident_rule['support']:.3f}\n\n")

    print("[INFO] insights.txt saved.")

    # -----------------------------
    # WRITE FULL REPORT
    # -----------------------------
    report_path = "output/reports/groceries_report.txt"

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("GROCERIES MARKET BASKET ANALYSIS REPORT\n")
        f.write("=======================================\n\n")

        f.write(f"Total Rules Generated: {len(rules)}\n\n")

        f.write("STRONGEST RULE (HIGHEST LIFT):\n")
        f.write(f"  {list(strongest_rule['antecedents'])} -> {list(strongest_rule['consequents'])}\n")
        f.write(f"  Lift: {strongest_rule['lift']:.4f}\n")
        f.write(f"  Confidence: {strongest_rule['confidence']:.4f}\n\n")

        f.write("MOST CONFIDENT RULE:\n")
        f.write(f"  {list(most_confident_rule['antecedents'])} -> {list(most_confident_rule['consequents'])}\n")
        f.write(f"  Confidence: {most_confident_rule['confidence']:.4f}\n")
        f.write(f"  Support: {most_confident_rule['support']:.4f}\n\n")

        f.write("INSIGHTS:\n")
        f.write("- Items with high lift strongly influence each other.\n")
        f.write("- Confidence shows reliability of the rule.\n")
        f.write("- Support indicates how often items appear.\n")

    print(f"[INFO] Full text report saved at: {report_path}")

    return {
        "total_rules": len(rules),
        "strongest_rule": strongest_rule,
        "most_confident_rule": most_confident_rule,
    }


