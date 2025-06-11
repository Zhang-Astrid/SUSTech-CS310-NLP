import os
import json

def count_tokens(text):
    # 简单分词，适合中英文混合文本
    return len(text.split())

def analyze_json_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        # 支持单条或多条样本
        if isinstance(data, dict):
            data = [data]
        total = len(data)
        label_count = {}
        total_tokens = 0
        samples = []
        for i, item in enumerate(data):
            text = item.get('text', '')
            label = item.get('label', 'N/A')
            label_count[label] = label_count.get(label, 0) + 1
            tokens = count_tokens(text)
            total_tokens += tokens
            if i < 3:
                samples.append({'text': text[:50], 'label': label, 'tokens': tokens})
        avg_tokens = total_tokens / total if total else 0
        print(f"文件: {file_path}")
        print(f"样本总数: {total}")
        print(f"标签分布: {label_count}")
        print(f"平均token数: {avg_tokens:.1f}")
        print("样本结构预览:")
        for s in samples:
            print(s)

def analyze_dir(root_dir, label_expected):
    domain_stats = {}
    for domain in os.listdir(root_dir):
        domain_path = os.path.join(root_dir, domain)
        if not os.path.isdir(domain_path):
            continue
        total_count = 0
        total_tokens = 0
        all_samples = []
        for fname in os.listdir(domain_path):
            if not fname.endswith('.json'):
                continue
            file_path = os.path.join(domain_path, fname)
            analyze_json_file(file_path)
            total_count += 1
            total_tokens += total_tokens
            all_samples.extend(samples)
        domain_stats[domain] = {
            'count': total_count,
            'avg_tokens': total_tokens / total_count if total_count else 0,
            'samples': all_samples
        }
    return domain_stats

def main():
    # base_dir = 'face2_zh_json'
    # human_dir = os.path.join(base_dir, 'human')
    # gen_dir = os.path.join(base_dir, 'generated')

    print("分析人工数据（label=0）...")
    human_stats = analyze_dir('Project/face2_zh_json/human/zh_unicode/news-zh.json', label_expected=0)
    print("分析AI生成数据（label=1）...")
    gen_stats = analyze_dir('Project/face2_zh_json/generated/zh_qwen2/news-zh.qwen2-72b-base.json', label_expected=1)

    print("\n=== 数据集统计信息 ===")
    for domain in sorted(set(human_stats) | set(gen_stats)):
        h = human_stats.get(domain, {'count': 0, 'avg_tokens': 0, 'samples': []})
        g = gen_stats.get(domain, {'count': 0, 'avg_tokens': 0, 'samples': []})
        print(f"\nDomain: {domain}")
        print(f"  人工样本数: {h['count']}, 平均token数: {h['avg_tokens']:.1f}")
        print(f"  AI样本数: {g['count']}, 平均token数: {g['avg_tokens']:.1f}")
        print("  人工样本预览:", h['samples'])
        print("  AI样本预览:", g['samples'])

if __name__ == "__main__":
    # 替换为你要分析的json文件路径
    json_path = "face2_zh_json/human/xxx.json"
    analyze_json_file(json_path) 