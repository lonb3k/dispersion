# Guide de présentation : dispersion

## 1. Les concepts

### L'identité de panier
σ²index = ρ (Σ wᵢ σᵢ)² + (1 − ρ) Σ wᵢ² σᵢ². Une seule corrélation moyenne ρ relie la vol de
l'indice aux vols des composants. Avec des vols implicites on obtient la **corrélation
implicite**, avec des vols réalisées la **corrélation réalisée**. La vol de l'indice est
bornée par la moyenne pondérée des vols single (ρ = 1) ; l'écart est la dispersion.

### Pourquoi la corrélation implicite est au-dessus de la réalisée
Les options sur indice sont achetées comme assurance de portefeuille, les options single-name
sont vendues par les overwriters. L'indice est donc cher relativement à ses composants, ce qui
se traduit par une corrélation implicite trop haute. C'est le **correlation risk premium** :
depuis 2015, +12 points en moyenne, positif 84 % des jours.

### Le trade de dispersion
Short variance indice, long variances single-name, pondéré en vega. Au premier ordre c'est
une position short corrélation plus un pari sur les primes de variance single. Le
**correlation swap** isole la jambe corrélation : vendre au strike COR1M, recevoir la
corrélation réalisée. Backtest mensuel : gagnant 86 % des mois, pire perte février 2020.

### Pourquoi la corrélation implicite est à un plus bas historique
Les Mag 7 pèsent 54 % du panier et 66 % de sa variance, mais leur corrélation entre eux (20 %)
et avec le reste (5 %) est faible : chacun trade sur sa propre histoire. La part de variance
expliquée par le premier facteur (PCA) est tombée de 40-45 % en 2020-2022 à 20 %. Cette part
et COR1M bougent ensemble (+0,60 de corrélation). Facteur marché faible = indice bon marché =
corrélation implicite basse. L'effondrement du premium en 2024-2025 est structurel, pas un
hasard.

### Short corrélation = short put
La corrélation monte exactement quand l'indice baisse. Régression du P&L mensuel du swap sur
le rendement S&P : pente positive, plus raide à la baisse. Dans les mois où l'indice perd plus
de 3 %, le swap perd 2 points contre +12 en moyenne. Vendre de la corrélation, c'est vendre
de l'assurance contre le krach avec une autre étiquette.

### Saisonnalité earnings
41 des 50 noms publient dans les 30 jours fin octobre : la vol single est gonflée par les
résultats, pas encore réalisée, donc la corrélation implicite est écrasée. La dispersion est
chère à vendre fin octobre et fin janvier.

## 2. Ce que fait le code

- `universe.py` : fichier des positions SPY (State Street), top 50 par poids, repondéré.
- `data.py` : prix yfinance (cache par date de départ), historique CBOE COR1M/COR3M/VIX
  depuis 2006 par le CSV public du CBOE, chaînes d'options.
- `options.py` : même moteur que le dashboard vol (forward par parité, Black-76).
- `snapshot.py` : 50 chaînes + SPY, IV ATM par nom, résolution de l'identité → ρ implicite.
  Fichier du jour réutilisé pour éviter deux minutes de lecture à chaque relance.
- `correlation.py` : corrélation réalisée glissante. **Convention** : la jambe indice est la
  vol de SPY (indice complet), comme le CBOE. Avec la vol du panier top-50 on obtient une
  corrélation plus haute et un faux premium négatif, c'est l'erreur qu'on a corrigée.
  Les noms sans historique (pré-IPO) sortent et les poids sont renormalisés chaque jour.
- `backtest.py` : premium quotidien, swap mensuel non chevauchant, stats, tables par année et
  par régime de VIX.
- `tradesheet.py` : livre vega-weighted, P&L(ρ, k) où k = réalisée/implicite des single names,
  grille de scénarios, corrélation de breakeven par bissection, P&L aux vols trailing.
- `factors.py` : corrélations intra Mag 7 / intra reste / croisées, parts de variance, part du
  premier facteur (plus grande valeur propre de la matrice de corrélation / n).
- `convexity.py` : put 15 delta vendu au VIX, régression P&L = a + b r + c min(r, 0).

## 3. Questions probables

**Survivorship ?** Le panier réalisé est le top 50 d'aujourd'hui. Avant 2015, NVDA, TSLA, PLTR
n'y étaient pas, donc la corrélation réalisée ancienne est sous-estimée et le premium
surestimé. Le backtest démarre en 2015 par défaut, et c'est écrit dans l'app.

**Pourquoi SPY et pas le panier comme jambe indice ?** Parce que le CBOE fait comme ça et parce
que l'indice complet est plus diversifié que 50 noms : avec le panier, ρ est mécaniquement
plus haut.

**Pourquoi un correlation swap plutôt qu'un vrai livre de variance swaps ?** Parce que les
strikes de variance single-name historiques n'existent pas en données gratuites. Le swap isole
la jambe corrélation proprement ; le trade sheet montre le livre complet sur les données du
jour.

**Le Sharpe du swap est-il crédible ?** Non, c'est une borne haute : brut, sans bid-ask (plusieurs
points sur un vrai swap), panier figé. Ce qui est robuste : le signe du premium, sa saisonnalité
et sa convexité.

**Combien vaut votre réplication ?** 10,0 % contre 10,6 % pour COR1M le même jour, avec des
cotations retardées. La méthode est la même que le CBOE.

## 4. D'où vient la prime de variance du S&P (onglet "Where the premium comes from")

C'est l'analyse qui relie les trois projets. Le vol dashboard montre que vendre de la vol sur
l'indice rapporte. Ici on répond à la question suivante : **qui paie cette prime, et pour quoi ?**

### 4.1 La question, en une phrase

Le vendeur de variance indice encaisse VIX² et paie la variance que l'indice réalise ensuite.
L'écart moyen est la prime de variance. Or la variance de l'indice dépend de deux choses : la vol
de chaque titre et la corrélation entre eux. On veut savoir laquelle des deux le marché surpaie.

### 4.2 Pourquoi on travaille en variance et pas en vol

La formule du panier est exacte en variance, pas en vol. Et le produit qui se traite sur desk est le
variance swap, dont le strike est justement le VIX². Un point de variance, c'est une vol au carré :
un VIX à 20 vaut 400 points de variance.

### 4.3 L'outil : l'identité de panier, écrite deux fois

```
V(σ, ρ) = B + ρ × (A − B)
A = (Σ wᵢ σᵢ)²     le carré de la vol moyenne pondérée des titres
B = Σ wᵢ² σᵢ²      ce que vaudrait la variance si les titres étaient indépendants
```

On l'écrit une fois avec ce que le marché anticipe (côté implicite, noté Q), une fois avec ce qui s'est
passé sur les 21 jours suivants (côté réalisé, noté P). La prime totale, V(Q) − V(P), dépend de deux
écarts : l'écart de corrélation (ρ_Q − ρ_P) et l'écart de vols (A et B).

### 4.4 Comment on coupe la prime en deux : Shapley

Il faut répartir la prime entre les deux écarts. Si on change d'abord la corrélation puis les vols,
on n'obtient pas le même partage que dans l'ordre inverse, à cause du terme croisé. La décomposition de
Shapley prend la moyenne des deux ordres : c'est la seule répartition qui ne favorise aucune variable.

```
part corrélation = (ρ_Q − ρ_P) × ½ × [(A_Q − B_Q) + (A_P − B_P)]
part vol individuelle = prime totale − part corrélation
```

La somme des deux parts redonne exactement la prime, tous les jours. Un test le vérifie.

### 4.5 Le problème : il n'y a pas d'historique gratuit des vols implicites des 50 titres

Côté réalisé, tout se calcule à partir des prix. Côté implicite, on a le VIX (la variance de l'indice)
et le COR1M (la corrélation implicite) depuis 2006, mais pas les vols implicites des titres. On les
reconstruit : l'identité contient une seule inconnue, le niveau des vols individuelles. On garde la forme
des vols réalisées récentes (NVDA plus volatile que KO, dans les mêmes proportions) et on résout pour
le niveau.

### 4.6 Le piège, et comment on l'a évité

La première version branchait le COR1M directement dans le VIX. Contrôlée contre les vraies chaînes du
27 octobre, elle donnait une vol moyenne des titres de 46,7 alors que les chaînes donnent 41,6. Elle était
fausse de 12 %.

La raison : le COR1M est calculé avec des vols **à la monnaie** (50 delta), alors que le VIX intègre
**tout le smile**, donc la prime des puts de protection. Sur l'indice, cette prime est énorme : ce jour-là,
la vol à la monnaie du SPY valait 13,8 et sa variance model-free 17,5. Mélanger les deux revenait à mettre
la cherté du skew de l'indice sur le dos des titres individuels.

La version corrigée suit la construction du CBOE, en trois temps :
1. **Vol de l'indice à la monnaie** : c_I × VIX, avec c_I = 0,79 mesuré sur SPY.
2. **Vols des titres à la monnaie** : on les déduit de cette vol de l'indice et du COR1M, avec la même
   formule que le CBOE.
3. **Passage aux variance swaps** : on multiplie les vols des titres par c_S = 1,13, mesuré sur les 49
   chaînes (le smile des titres coûte moins cher que celui de l'indice). La corrélation implicite
   model-free se déduit ensuite du VIX².

Résultat du contrôle : 41,1 reconstruit contre 41,6 mesuré. Pour le calibrage, on a répliqué la variance
de chaque titre avec la même méthode que le VIX (le strip d'options du vol dashboard). Sur SPY, elle
redonne 17,5 pour un VIX à 17,7.

### 4.7 Les résultats

| Points de variance | 2015-2023 | 2024-2026 |
|---|---|---|
| Prime de l'indice | 70 | 84 |
| dont corrélation | +217 | +93 |
| dont vol individuelle | −147 | −9 |

Trois choses à retenir :

- **Avant 2024, toute la prime venait de la corrélation.** La part corrélation est positive 97 % des
  jours. La vol individuelle, elle, n'était pas surpayée, sa part est même négative. C'est le résultat de
  Driessen, Maenhout et Vilkov (Journal of Finance, 2009) : la prime de variance de l'indice est une
  prime de corrélation. On le retrouve avec des données gratuites.
- **La corrélation model-free est 7 à 12 points au-dessus du COR1M.** Le skew de l'indice est beaucoup
  plus raide que celui des titres. Quand on achète un put sur l'indice, on se protège surtout contre le
  scénario où tous les titres baissent ensemble, c'est-à-dire contre la corrélation qui monte vers 1.
- **Les pertes des pires mois viennent des vols individuelles, pas de la corrélation.** En mars 2020,
  ce sont les vols des titres qui explosent qui font perdre le vendeur de variance indice. La part
  corrélation est la source de revenu régulière.

### 4.8 Ce qui est robuste, et ce qui ne l'est pas

Deux choix pèsent sur les niveaux :
- **Le calibrage.** c_I et c_S ne sont mesurés que sur les jours où l'on a les chaînes. On les traite
  comme des constantes, et on teste trois couples : 0,75/1,05, 0,80/1,10 et 0,85/1,15.
- **Le panier.** Le top 50 d'aujourd'hui contient des survivants. NVDA, TSLA et AVGO étaient plus petits
  et beaucoup plus volatils avant 2024, ce qui gonfle la vol réalisée des titres et pousse
  artificiellement vers "tout vient de la corrélation". Le panier des 31 titres déjà dans le top 50 en
  2015 corrige ce biais, mais il en crée un autre après 2024, puisqu'il manque les titres qui font
  aujourd'hui la variance de l'indice.

Ce qui tient dans les six variantes : la part corrélation est positive avant 2024 ; après 2024 elle
baisse de 90 à 155 points, et la part vol individuelle monte de 105 à 170 points. Ce qui ne tient pas :
l'ampleur exacte du déplacement et le signe de la part individuelle. **On annonce un sens, pas un chiffre.**

### 4.9 Questions probables

**"Votre part corrélation, c'est le P&L d'un trade de dispersion ?"** Pas exactement. C'est une
attribution de la position courte en variance indice. Un vrai livre de dispersion paie en plus 50
fourchettes de prix sur les variance swaps des titres, ce qui mange une bonne partie de la prime. C'est
d'ailleurs l'argument de Driessen et al. : la prime de corrélation existe, mais elle est difficile à
capter après coûts.

**"Pourquoi Shapley et pas une décomposition simple ?"** Parce qu'il y a un terme croisé : vol et
corrélation bougent ensemble, surtout en crise. Sans Shapley, le partage dépend de l'ordre dans lequel on
fait varier les deux, ce qui est arbitraire.

**"Comment savez-vous que vos vols implicites reconstruites sont justes ?"** On les a comparées aux vraies
chaînes du 27 octobre : 41,1 contre 41,6. Et on a montré que la méthode naïve se trompait de 12 %. C'est
ce contrôle qui nous a fait corriger la méthode.

**"Pourquoi la prime se déplace vers les titres depuis 2024 ?"** Deux pistes compatibles avec les données
du projet. D'abord, l'indice est devenu quelques géants peu corrélés entre eux : le COR1M est à des plus
bas historiques, donc il reste peu de corrélation à surpayer. Ensuite, ces géants ont des événements
propres très chers, comme les earnings de NVDA ou les annonces IA, qui gonflent leur vol implicite. Ce
sont des hypothèses, pas des résultats démontrés.

**"Quelle est la plus grosse faiblesse ?"** Les deux constantes de calibrage, mesurées sur une seule
journée de chaînes. Le relevé quotidien des chaînes accumule maintenant un historique, ce qui permettra
de les mesurer dans le temps.

## 5. Pitch en trois phrases

"J'ai reconstruit l'indice de corrélation implicite du CBOE depuis 50 chaînes d'options, à
0,6 point près. J'ai mesuré le premium de corrélation depuis 2006 et montré qu'il s'est effondré
en 2024 parce que les Mag 7 se sont décorrélés : la part du facteur marché est passée de 45 % à
20 %. Et j'ai montré que vendre de la corrélation, c'est vendre un put sur l'indice, avec la
grille de scénarios qu'un desk regarderait avant de mettre le trade."
