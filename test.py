from tqdm.rich import tqdm
import time

# Exécution d'une boucle principale (1er niveau de progression)
for i in tqdm(range(5), desc="Main Loop"):
    # Exécution d'une boucle imbriquée (2nd niveau de progression)
    for j in tqdm(range(3), desc="Sub Loop", leave=False):
        time.sleep(0.5)  # Simuler une tâche

    # Simuler un délai supplémentaire pour le niveau principal
    time.sleep(1)
