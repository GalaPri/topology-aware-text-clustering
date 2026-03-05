import requests
import pandas as pd
from tqdm import tqdm
import time
import json

# ---- Settings ----
'''CLUSTERS = {
    "metabolic_basic": [
       "Metabolic syndrome", "Insulin resistance", "Type 2 diabetes", "BMI"
    ],
    "cardio_obvious": [
       "Cardiovascular disease", "Coronary artery disease", "Atherosclerosis", "Heart failure"
    ],
    "cardio_overlap": [
       "Endothelial dysfunction", "Oxidative stress", "Inflammation", "Proinflammatory state"
    ],
    "neuro_overlap": [
       "Neuroinflammation", "Brain metabolism", "neurodegeneration", "oxidative stress"
    ],
    "noise_mixed": [
       "Quantum computing", "Topological data analysis", "GAN neural networks"
    ]
}'''

'''CLUSTERS = {
    "ai_ml": [
        "Machine learning", "Deep learning", "Neural networks", "Reinforcement learning"
    ],
    "quantum_physics": [
        "Quantum computing", "Quantum entanglement", "Quantum teleportation", "Quantum algorithms"
    ],
    "robotics": [
        "Humanoid robots", "Industrial robotics", "Robot control systems", "Swarm robotics"
    ],
    "data_science": [
        "Data mining", "Big data analytics", "Data visualization", "Predictive analytics"
    ],
    "cryptography": [
        "Public key cryptography", "Zero-knowledge proofs", "Homomorphic encryption", "Blockchain security"
    ],
    "topology_math": [
        "Algebraic topology", "Persistent homology", "Simplicial complexes", "Topological data analysis"
    ],
    "computer_graphics": [
        "Ray tracing", "3D rendering", "Shader programming", "Procedural generation"
    ]
}

CLUSTER_SIZES = {
    "ai_ml": 800,              # was 2000 → maxed out
    "quantum_physics": 800,    # stays at 800
    "robotics": 600,           # was 1500 → scaled down
    "data_science": 800,       # was 2500 → maxed out
    "cryptography": 400,       # was 700 → scaled down
    "topology_math": 300,      # stays at 300
    "computer_graphics": 500   # was 1200 → scaled down
}'''

'''CLUSTERS = {
    "ai_data": [
        "Machine learning", "Deep learning", "Data mining", "Predictive analytics"
    ],
    "ml_robotics": [
        "Reinforcement learning", "Neural networks", "Robot control systems", "Humanoid robots"
    ],
    "quantum_theory": [
        "Quantum computing", "Quantum algorithms", "Quantum entanglement", "Quantum simulations"
    ],
    "crypto_security": [
        "Public key cryptography", "Blockchain security", "Zero-knowledge proofs", "Homomorphic encryption"
    ],
    "data_visualization": [
        "Data visualization", "Big data analytics", "Graph theory", "Interactive dashboards"
    ],
    "computer_graphics": [
        "3D rendering", "Shader programming", "Procedural generation", "Ray tracing"
    ]
}
CLUSTER_SIZES = {
    "ai_data": 700,
    "ml_robotics": 600,
    "quantum_theory": 500,
    "crypto_security": 400,
    "data_visualization": 600,
    "computer_graphics": 500
}'''

'''CLUSTERS = {
    "bioinformatics_ai": [
        "Genomic data", "Protein folding", "Machine learning", "Neural networks",
        "Predictive modeling", "Data mining"  # перекрытие с AI
    ],
    "neuro_robotics": [
        "Neural networks", "Brain-computer interface", "Cognitive robotics", "Motor control",
        "Reinforcement learning"  # перекрытие с AI/ML
    ],
    "quantum_crypto": [
        "Quantum computing", "Quantum key distribution", "Quantum algorithms",
        "Blockchain security", "Homomorphic encryption"  # перекрытие с crypto
    ],
    "eco_geoinfo": [
        "Climate modeling", "Remote sensing", "GIS", "Species distribution",
        "Big data analytics"  # перекрытие с data analysis
    ],
    "graphics_vrar": [
        "3D rendering", "Shader programming", "Ray tracing", "Virtual reality",
        "Augmented reality", "Interactive dashboards"  # перекрытие с data visualization
    ],
    "econ_social": [
        "Economic forecasting", "Social network analysis", "Predictive analytics",
        "Big data analytics", "Trend prediction"  # перекрытие с data analysis
    ]
}

CLUSTER_SIZES = {
    "bioinformatics_ai": 600,
    "neuro_robotics": 500,
    "quantum_crypto": 450,
    "eco_geoinfo": 550,
    "graphics_vrar": 500,
    "econ_social": 200
}'''


#ok im getting worried now
'''CLUSTERS = {
    # AI и Bioinformatics пересекаются по ML и анализу данных
    "bioinformatics_ai": [
        "Machine learning", "Deep learning", "Genomics", "Protein folding", 
        "Gene expression", "Neural networks", "Predictive analytics"
    ],
    # Robotics и Neuroscience пересекаются по когнитивным и биомеханическим системам
    "neuro_robotics": [
        "Neural networks", "Brain-computer interface", "Motor control", 
        "Humanoid robots", "Neuroprosthetics", "Cognitive robotics"
    ],
    # Экономика и геоинформатика пересекаются по моделированию и анализу пространственных данных
    "eco_geoinfo": [
        "Spatial analysis", "Geographic information systems", "Economic modeling", 
        "Remote sensing", "Urban planning", "Big data analytics"
    ],
    # Graphics и VR/AR пересекаются с AI по компьютерной визуализации
    "graphics_vrar": [
        "3D rendering", "Shader programming", "Virtual reality", "Augmented reality", 
        "Procedural generation", "Deep learning"
    ],
    # Квантовая криптография, Quantum Computing пересекается с AI по оптимизации и алгоритмам
    "quantum_crypto": [
        "Quantum computing", "Quantum algorithms", "Quantum entanglement", 
        "Blockchain security", "Public key cryptography"
    ],
    # Социальная экономика с шумными пересечениями
    "econ_social": [
        "Social networks", "Economic modeling", "Predictive analytics", 
        "Behavioral economics", "Data visualization"
    ]
}

CLUSTER_SIZES = {
    "bioinformatics_ai": 600,
    "neuro_robotics": 500,
    "eco_geoinfo": 550,
    "graphics_vrar": 500,
    "quantum_crypto": 450,
    "econ_social": 200
}'''
'''CLUSTERS = {
    "ai_quantum": [
        "Machine learning", "Quantum computing", "Neural networks", "Quantum algorithms"
    ],
    "robotics_ai": [
        "Humanoid robots", "Reinforcement learning", "Industrial robotics", "Neural networks"
    ],
    "bioinformatics_ai": [
        "Genome sequencing", "Machine learning", "Protein folding", "Data mining"
    ],
    "quantum_crypto": [
        "Quantum entanglement", "Blockchain security", "Quantum algorithms", "Homomorphic encryption"
    ],
    "graphics_vrar": [
        "3D rendering", "Shader programming", "VR/AR applications", "Machine learning"
    ],
    "econ_social": [
        "Big data analytics", "Predictive analytics", "Social network analysis", "Quantum computing"
    ]
}

CLUSTER_SIZES = {
    "ai_quantum": 500,
    "robotics_ai": 450,
    "bioinformatics_ai": 400,
    "quantum_crypto": 350,
    "graphics_vrar": 300,
    "econ_social": 250
}
'''
'''CLUSTERS = {
    "ai_quantum": [
        "Machine learning", "Quantum computing", "Neural networks", "Quantum algorithms"
    ],
    "robotics_ai": [
        "Humanoid robots", "Reinforcement learning", "Industrial robotics", "Neural networks"
    ],
    "bioinformatics_ai": [
        "Genome sequencing", "Machine learning", "Protein folding", "Data mining"
    ],
    "quantum_crypto": [
        "Quantum entanglement", "Blockchain security", "Quantum algorithms", "Homomorphic encryption"
    ],
    "graphics_vrar": [
        "3D rendering", "Shader programming", "VR/AR applications", "Machine learning"
    ],
    "econ_social": [
        "Big data analytics", "Predictive analytics", "Social network analysis", "Quantum computing"
    ],
    "ml_finance": [
        "Machine learning", "Stock prediction", "Risk analysis", "Neural networks"
    ],
    "robotics_quantum": [
        "Quantum sensors", "Humanoid robots", "Reinforcement learning", "Quantum computing"
    ],
    "bioinformatics_quantum": [
        "Protein folding", "Quantum algorithms", "Genome sequencing", "Data mining"
    ],
    "vrar_crypto": [
        "VR/AR applications", "Blockchain security", "3D rendering", "Homomorphic encryption"
    ]
}

CLUSTER_SIZES = {
    "ai_quantum": 150,
    "robotics_ai": 150,
    "bioinformatics_ai": 120,
    "quantum_crypto": 100,
    "graphics_vrar": 120,
    "econ_social": 100,
    "ml_finance": 120,
    "robotics_quantum": 130,
    "bioinformatics_quantum": 100,
    "vrar_crypto": 100
}'''
CLUSTERS = {
    "mega_cluster": [
        "Machine learning", "Deep learning", "Data mining", "Predictive analytics",
        "Reinforcement learning", "Neural networks", "Robot control systems", "Humanoid robots",
        "Quantum computing", "Quantum algorithms", "Quantum entanglement", "Quantum simulations",
        "Public key cryptography", "Blockchain security", "Zero-knowledge proofs", "Homomorphic encryption",
        "Data visualization", "Big data analytics", "Graph theory", "Interactive dashboards",
        "3D rendering", "Shader programming", "Procedural generation", "Ray tracing"
    ],
    "tiny_cluster_1": [
        "Bioinformatics", "Genomics", "Proteomics"
    ],
    "tiny_cluster_2": [
        "Climate change", "Geoinformatics", "Environmental modeling"
    ]
}

CLUSTER_SIZES = {
    "mega_cluster": 3000,
    "tiny_cluster_1": 100,
    "tiny_cluster_2": 80
}
'''CLUSTERS = {
    "ai_quantum_mashup": [
        "Machine learning", "Quantum computing", "Neural networks", "Quantum algorithms"
    ],
    "bioinformatics_graphics": [
        "Bioinformatics", "3D rendering", "Genome analysis", "Shader programming"
    ],
    "crypto_robotics": [
        "Blockchain security", "Humanoid robots", "Reinforcement learning", "Homomorphic encryption"
    ],
    "eco_neuro": [
        "Neurodegeneration", "Climate modeling", "Brain metabolism", "Environmental simulation"
    ],
    "mixed_noise": [
        "GAN networks", "Topological data analysis", "Quantum entanglement", "VRAR simulations",
        "Predictive analytics", "Big data visualization", "AI ethics", "Industrial robotics"
    ]
}

CLUSTER_SIZES = {
    "ai_quantum_mashup": 700,
    "bioinformatics_graphics": 500,
    "crypto_robotics": 600,
    "eco_neuro": 400,
    "mixed_noise": 300
}'''

MAX_PAPERS_TOTAL = 10000
SAVE_EVERY = 1000

# ---- Containers ----
global_results = []  # will be saved to JSON at the end
count_total = 0

# ---- Main Loop ----
for cluster, keywords in CLUSTERS.items():

    cluster_results = []  # separate for Excel per cluster
    max_cluster = CLUSTER_SIZES[cluster]
    count_cluster = 0

    for kw in tqdm(keywords, desc=f"Cluster: {cluster}"):

        if count_total >= MAX_PAPERS_TOTAL:
            break

        url = f"https://api.openalex.org/works?search={kw}&per-page=200"
        cursor = None

        while url and count_total < MAX_PAPERS_TOTAL:
            response = requests.get(url)
            data = response.json()

            for item in data["results"]:
                if count_total >= MAX_PAPERS_TOTAL:
                    break

                # Abstract conversion
                abstract_dict = item.get("abstract_inverted_index")
                abstract = None
                if abstract_dict:
                    abstract = " ".join(abstract_dict.keys())

                authors = ", ".join(
                            a.get("author", {}).get("display_name") 
                            for a in item["authorships"] 
                            if a.get("author", {}).get("display_name")
                        ) or "Unknown"

                affiliations = "; ".join(
                            ", ".join(inst.get("display_name") for inst in a.get("institutions", []) if inst.get("display_name"))
                            for a in item.get("authorships", [])
                        ) or "Unknown"


                record = {
                    "cluster": cluster,  # <---- NEW
                    "keyword": kw,
                    "title": item.get("title"),
                    "abstract": abstract,
                    "authors": authors,
                    "affiliations": affiliations,
                    "year": item.get("publication_year"),
                    "journal": item.get("host_venue", {}).get("display_name"),
                    "doi": item.get("doi"),
                    "citations": item.get("cited_by_count")
                }
                if count_cluster >= max_cluster or count_total >= MAX_PAPERS_TOTAL:
                    break
                global_results.append(record)
                cluster_results.append(record)
                count_total += 1
                count_cluster += 1 

            # Pagination
            cursor = data["meta"].get("next_cursor")
            if cursor:
                url = f"https://api.openalex.org/works?search={kw}&cursor={cursor}&per-page=200"
            else:
                url = None

            time.sleep(1)

    # ---- Save cluster Excel ----
    df_cluster = pd.DataFrame(cluster_results)
    df_cluster.to_excel(f"{cluster}.xlsx", index=False)
    print(f"📁 Saved {cluster}.xlsx with {len(df_cluster)} records")

# ---- Final Save: JSON ----
with open("dirty.json", "w", encoding="utf-8") as f:
    json.dump(global_results, f, ensure_ascii=False, indent=2)

print(f"Total papers saved: {count_total}")
