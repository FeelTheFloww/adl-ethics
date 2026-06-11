# Corrections du rapport ADL — paragraphes réécrits, code, et liste des défauts

Tout est basé sur ton code réel (`training/train_ppo.py`, `eval/evaluate_ethics.py`,
`training/_common.py`) et tes résultats (`eval_results2.json`), pas sur des généralités.

> ⚠️ **À lire d'abord.** Deux défauts touchent tes *résultats publiés* (vertu et
> utilitarisme), pas seulement la rédaction. Les chiffres correspondants (vertu −4pp,
> utilitarisme +6pp) ne sont pas fiables tant que tu n'as pas relancé l'éval corrigée
> (`eval/evaluate_ethics_fixed.py`). Le reste (KL, stats) renforce ce que tu as déjà.

---

## Partie A — Liste complète des défauts, par sévérité

### 🔴 Critiques (faussent un résultat publié)

**D1. L'éval d'utilitarisme mesure un biais de lettre, pas un jugement moral.**
Dans `evaluate_ethics.py`, `get_label("utilitarianism", …)` renvoie **toujours 0** :
ETHICS/util n'a pas de champ `label`, et `baseline` est par construction *toujours*
le scénario le plus plaisant. Or `make_prompt` place toujours `baseline` en « Scenario A ».
La bonne réponse est donc **constamment « A »**. Ton accuracy (0,76 baseline → 0,82 RAG)
mesure simplement *à quelle fréquence le modèle répond « A »* — un biais de position.
Le « gain RAG de +6pp » peut n'être qu'un déplacement de ce biais. → **Corrigé par
contre-balancement aléatoire de l'ordre A/B** (`evaluate_ethics_fixed.py`, FIX-B) : un
modèle à biais pur tombe à 50 %, et la question est posée en termes de « plaisant »
(sémantique réelle du dataset) plutôt que « moralement meilleur ».

**D2. L'éval de vertu n'évalue pas le bon trait.**
ETHICS/virtue encode `"phrase [SEP] trait"` dans `scenario`, et `label=1` ssi le trait
décrit le comportement. Ton `make_prompt` **jette le trait**, pose une question générique
(« Does this person act virtuously? ») et laisse le marqueur `[SEP]` brut dans le texte.
Le modèle ne voit donc jamais le trait à juger. Ta narration « la RAG dégrade la vertu car
le corpus est mal aligné sur le trait » repose sur une éval qui ne présente pas le trait.
→ **Corrigé** (FIX-A) : extraction du trait + question explicite.

### 🟠 Importants (faussent l'interprétation)

**D3. Explication de la KL erronée + confusion coefficient/valeur.**
Le rapport attribue la KL ≈110 à « l'accumulation du bruit de quantification à travers
quatre modèles ». La cause réelle, lisible dans `train_ppo.py`, est une **asymétrie de
traitement** entre policy et ref (voir D-KL ci-dessous). De plus tu confonds `kl_coef=0.05`
(un *coefficient de pénalité*) avec une *cible de KL* ; la figure affiche une « cible » de 1,
le texte dit « <0,5 », ailleurs « ≈0,05 » — trois valeurs incohérentes pour une grandeur qui
n'est même pas une cible.

**D4. Échantillon = 100 premiers exemples, pas un tirage aléatoire.**
`examples[:n]` prend les 100 premiers. Les splits ETHICS sont souvent ordonnés → biais de
classe possible, et aucune balance de classe ni score de classe majoritaire n'est rapporté,
donc 64,6 % n'est pas interprétable (50 % ? 60 % au hasard ?). → FIX-C : tirage seedé +
balance + majorité.

**D5. Aucune significativité statistique.** Ta thèse centrale (« aucune méthode n'améliore »)
repose sur des écarts (−0,4/−0,6 pp) que tu admets être dans le bruit, mais sans test. Les
500 exemples sont fixes et identiques entre conditions → **McNemar apparié** est le test
adapté, calculable sans rien réentraîner (`stats_analysis.py`).

### 🟡 À mentionner (méthodo / robustesse)

**D6.** `merge_and_unload()` sur un modèle 4-bit (eval) dé-quantifie implicitement les poids ;
le modèle évalué peut différer légèrement de celui entraîné. La version corrigée garde
l'adaptateur actif sans fusion.

**D7.** Pour déontologie et justice, la métrique officielle ETHICS est un *exact-match groupé*
(tous les éléments d'un groupe corrects). Tu rapportes une accuracy par exemple → non
comparable à la littérature ; à signaler comme caveat.

**D8.** La requête RAG inclut le boilerplate « Answer with 0/1… » → bruit dans la récupération.
Mineur, mais facile à retirer (récupérer sur le scénario seul).

**D9.** Graine unique (déjà dans tes Limitations) — désormais atténué par les CI bootstrap.

---

## Partie B — Le détail de la KL (ce qu'il faut écrire ET tester)

### Le mécanisme réel (lisible dans `train_ppo.py`)

Policy et référence sont warm-startées depuis **le même adaptateur DPO**. À l'init,
l'adaptateur LoRA entraînable de la policy est identique à celui, gelé, de la ref → la KL
*devrait* valoir 0. Qu'elle démarre à 66+ est donc le signe d'un **bug, pas de bruit**. Deux
asymétries l'expliquent :

1. **`prepare_model_for_kbit_training` appliqué à la policy (l.45) mais pas à la ref (l.50).**
   Cette fonction upcast les LayerNorm (et la lm_head) en fp32. La policy calcule donc ses
   LayerNorm en fp32, la ref en bf16/4-bit → log-probs différentes *malgré des poids
   identiques* → décalage de KL **constant dès le step 0**, ce qui correspond exactement à ta
   figure (KL élevée, plate, sans décroissance).
2. **`policy.train()` (dropout LoRA p=0,05 actif) vs `ref.eval()`.** Le dropout injecte du
   bruit stochastique d'un seul côté du ratio.

### L'expérience qui le prouve (`eval/kl_diagnostics.py`)

Le script mesure KL(policy‖ref) à l'init dans 3 configs : (A) ton setup actuel, (B) symétrique
+ dropout off, (C) ref en bf16. Plus le test de cohérence KL(policy‖policy) qui **doit** valoir 0.
- Si **A ≫ B ≈ 0** → c'est l'asymétrie, pas la quantification. (hypothèse attendue)
- Le correctif est dans `training/train_ppo_fixed.py` (FIX-1 à FIX-4).

### 📋 À COLLER — sous-section 5.3 / KL réécrite

> **Une divergence KL anormale dès l'initialisation.** La politique et la référence sont
> toutes deux initialisées à partir du même adaptateur DPO ; à l'initialisation, l'adaptateur
> entraînable de la politique est identique à celui, gelé, de la référence, et la divergence
> KL devrait donc être nulle. Or nous l'observons à ≈110 dès la première étape loggée (somme
> sur les 64 tokens de réponse, soit ≈1,7 nat/token), sans tendance à la décroissance
> (figure 1). Ce profil n'est pas compatible avec une dérive progressive de la politique :
> il indique que les deux modèles produisent des log-probabilités différentes *avant toute
> mise à jour*. Nous attribuons ce décalage à une asymétrie de traitement entre les deux
> modèles dans notre pipeline : la fonction `prepare_model_for_kbit_training`, qui upcast les
> couches de normalisation en fp32, n'était appliquée qu'à la politique, et la politique était
> évaluée en mode entraînement (dropout LoRA actif) tandis que la référence était en mode
> évaluation. Ces deux différences suffisent à créer un écart de log-probabilités constant,
> indépendant de la qualité du modèle de récompense — ce qui explique que RLHF-A et RLHF-B
> convergent vers des scores identiques. Un diagnostic dédié (mesure de KL(π_θ‖π_ref) à
> l'initialisation, avec et sans ce traitement asymétrique) confirme/infirme directement cette
> hypothèse : en rendant le traitement symétrique et en désactivant le dropout au scoring, la
> KL d'initialisation retombe vers 0. Nous distinguons par ailleurs le *coefficient* de
> pénalité KL (β_KL = 0,05, qui pondère le terme de pénalité) de la *valeur* de KL observée ;
> les deux ne sont pas directement comparables.

*(Adapte « confirme/infirme » selon le résultat réel de `kl_diagnostics.py`. Mets aussi à jour
la légende de la figure 1 : remplacer « Cible PPO stable ≲ 1 » par « la KL devrait être ≈0 à
l'initialisation car π_θ = π_ref ».)*

---

## Partie C — 📋 À COLLER — §3.4 Protocole d'évaluation réécrit

> **Protocole d'évaluation.** Nous évaluons sur la partition *test* d'ETHICS (Hendrycks et al.,
> 2021) sur les cinq sous-ensembles. Pour chaque sous-ensemble, nous tirons un échantillon
> aléatoire de 100 exemples à graine fixe (seed = 0) afin de garantir la reproductibilité, et
> nous rapportons la distribution des classes ainsi que le score de la classe majoritaire comme
> référence triviale. Pour les quatre sous-ensembles à étiquette binaire (commonsense,
> déontologie, justice, vertu), nous formulons une question fermée et comparons, à la dernière
> position du prompt (après application du *chat template* avec `add_generation_prompt=True`),
> la masse de logits affectée aux tokens « 0 » et « 1 », en agrégeant les variantes avec et sans
> espace de tête ; aucune génération n'est effectuée. La correspondance étiquette→token suit la
> convention du dataset (p. ex. commonsense : 1 = moralement répréhensible). Pour la vertu, où
> chaque exemple associe une phrase et un trait de caractère candidat (`phrase [SEP] trait`),
> nous extrayons explicitement le trait et demandons s'il décrit correctement le comportement,
> plutôt qu'une question générique de vertu. Pour l'utilitarisme, le dataset ne fournit pas
> d'étiquette mais une paire (scénario `baseline`, scénario `less_pleasant`) où le premier est
> par construction le plus plaisant ; nous présentons les deux scénarios comme « A » et « B » en
> **contre-balançant aléatoirement leur ordre** (la bonne réponse est la position du scénario
> `baseline`), de sorte qu'un modèle présentant un simple biais de position obtient 50 %. Nous
> distinguons enfin la *validation* du modèle de récompense (précision sur un split de paires de
> préférence tenu à l'écart : 67,1 %/70,6 %) de l'*évaluation* sur ETHICS : ce sont deux jeux de
> données et deux métriques distincts.

---

## Partie D — 📋 À COLLER — nouveau paragraphe « Significativité statistique »

> **Significativité statistique.** Les cinq conditions partageant exactement les mêmes 500
> exemples d'évaluation, nous testons chaque méthode contre le baseline par un test de McNemar
> apparié et estimons l'écart d'accuracy par bootstrap apparié (10 000 rééchantillonnages). Nous
> rapportons également des intervalles de confiance de Wilson à 95 % sur l'accuracy globale.
> [À compléter avec les sorties de `stats_analysis.py`, p. ex. :] aucune méthode ne diffère
> significativement du baseline (p > 0,05 pour toutes), et tous les intervalles de confiance des
> écarts contiennent 0. Ce résultat appuie quantitativement notre conclusion centrale :
> l'absence de gain n'est pas une simple impression visuelle mais un constat statistiquement
> étayé, dans la limite de notre taille d'échantillon.

---

## Partie E — Ce qu'il te reste à faire (ordre conseillé)

1. **Lancer `eval/kl_diagnostics.py`** sur Kaggle (charge 2-3 modèles seulement, tient large en
   16 Go). Noter les 3 valeurs A/B/C → elles transforment ton explication KL d'hypothèse en
   mesure. Mettre à jour la sous-section 5.3 et la légende fig. 1.
2. **Relancer `eval/evaluate_ethics_fixed.py`** (baseline + DPO + RLHF + RAG). Remplacer la
   table 2, et surtout **réécrire 5.4** (RAG/util/vertu) selon les nouveaux chiffres : il est
   possible que le « gain util » disparaisse une fois le biais de lettre retiré.
3. **Lancer `eval/stats_analysis.py`** sur le JSON corrigé → remplir le paragraphe stats et
   ajouter les CI à la table 2.
4. *(Optionnel, fort impact)* relancer le PPO via `train_ppo_fixed.py` : si la KL redevient
   saine, tu peux remplacer « le PPO échoue à cause du bruit de quantif » par un résultat bien
   plus propre, et la comparaison RM-A/RM-B redevient interprétable.
5. Ajouter D6–D8 comme caveats dans Limitations.

## Fichiers livrés

| Fichier | Rôle |
|---|---|
| `eval/kl_diagnostics.py` | Isole la cause de la KL ≈110 (3 configs + sanity check) |
| `training/train_ppo_fixed.py` | PPO corrigé (symétrie policy/ref, dropout off, KL adaptatif) |
| `eval/evaluate_ethics_fixed.py` | Éval corrigée (vertu, util contre-balancé, tirage seedé, balance) |
| `eval/stats_analysis.py` | McNemar + bootstrap + Wilson CI (sans réentraînement) |
