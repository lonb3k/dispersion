# Guide de présentation : dispersion

## 1. Les concepts

### L'identité de panier
σ²index = ρ (Σ wᵢ σᵢ)² + (1 − ρ) Σ wᵢ² σᵢ². Une seule corrélation moyenne ρ relie la vol de
l'indice aux vols des composants. Avec des vols implicites on obtient la **corrélation
implicite**, avec des vols réalisées la **corrélation réalisée**. La vol de l'indice est
bornée par la moyenne pondérée des vols single (ρ = 1) ; l'écart est la dispersion.

### La prime de corrélation, et le piège des survivants
L'explication classique : les options sur indice sont achetées comme assurance de portefeuille,
les options single-name sont vendues par les overwriters, donc la corrélation implicite serait trop
haute. Avec le top 50 d'aujourd'hui comme panier, on trouve en effet +13 points par mois depuis 2015.

**Ce chiffre est faux, et c'est le résultat le plus important du projet.** Les gagnants d'aujourd'hui
(NVDA, TSLA, AVGO, PLTR) étaient il y a dix ans des titres plus petits, plus volatils et peu corrélés au
marché. Le vrai top 50 de 2015 contenait XOM, PG, VZ ou Citi. Mesurer la corrélation passée sur les
survivants la sous-estime, donc fabrique une prime. On reconstruit le panier **point-in-time** : chaque
mois, les 50 plus grosses capitalisations du moment, estimées avec le nombre d'actions d'aujourd'hui
(fichier SPY) multiplié par le prix de l'époque.

Preuve que c'est bien le biais : en 2026, les deux paniers ont 47 titres en commun et donnent la même
corrélation réalisée (9,0 % contre 9,3 %). En 2015 ils n'en partagent que 29. En 2006, le panier figé
donne 0,2 % de corrélation réalisée, ce qui est absurde, contre 26,7 %.

Sur le panier point-in-time, depuis 2015 : COR1M moyen 28,6 %, corrélation réalisée le mois suivant
28,4 %. **À la monnaie, la corrélation implicite est une prévision presque sans biais.** Elle contient
une prime seulement en période de stress (+7,9 points quand le VIX dépasse 30), et elle est même
en dessous du réalisé quand le marché est calme (−2,8 points sous un VIX de 15).

### Le trade de dispersion
Short variance indice, long variances single-name, pondéré en vega. Au premier ordre c'est
une position short corrélation plus un pari sur les primes de variance single. Le
**correlation swap** isole la jambe corrélation : vendre au strike COR1M, recevoir la
corrélation réalisée. Backtest mensuel depuis 2015 sur le panier point-in-time : +0,4 point par mois,
gagnant 57 % des mois, Sharpe brut 0,1. Il ne rapporte rien, pour la raison ci-dessus.

### Pourquoi la corrélation implicite est à un plus bas historique
Les Mag 7 pèsent 54 % du panier et 66 % de sa variance, mais leur corrélation entre eux (20 %)
et avec le reste (5 %) est faible : chacun trade sur sa propre histoire. La part de variance
expliquée par le premier facteur (PCA) est tombée de 40-45 % en 2020-2022 à 20 %. Cette part
et COR1M bougent ensemble (+0,60 de corrélation). Facteur marché faible = indice bon marché =
corrélation implicite basse.

### Short corrélation = short put
La corrélation monte exactement quand l'indice baisse. Régression du P&L mensuel du swap sur
le rendement S&P : +2,1 points par 1 % de mouvement de l'indice, 0,5 de plus à la baisse. Dans les
21 mois où l'indice perd plus de 3 %, le swap perd 17 points contre +0,4 en moyenne. Vendre de la
corrélation, c'est vendre de l'assurance contre le krach avec une autre étiquette.

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
- `universe.py`, `data.py` : univers point-in-time. Chaque fin de mois, top 50 par capitalisation
  estimée (actions détenues par SPY aujourd'hui × prix de l'époque, ajusté des splits mais pas des
  dividendes), appliqué à partir du jour de bourse suivant. Si Yahoo refuse les requêtes, retour aux
  fichiers commités.
- `decomposition.py` : décomposition de Shapley de la prime de variance (section 4).
- `dispersion_backtest.py` : le livre vega-weighted en variance swaps, roulé chaque mois. Notionnel
  de variance = vega / (2 × strike), payoff = notionnel × (RV² − K²), titres plafonnés à 2,5 fois le
  strike, demi-fourchette payée à chaque roll. Pour passer en rendement, une taille constante qui fait
  tourner la stratégie à 10 % de vol : le Sharpe n'en dépend pas, le CAGR et le drawdown si.

## 3. Questions probables

**Survivorship ?** C'est le cœur du projet. Avec le top 50 d'aujourd'hui, on trouve une prime de
corrélation de 13 points par mois et un Sharpe de 3,3. Avec un panier point-in-time, il ne reste rien à
la monnaie. L'app permet de basculer entre les deux paniers pour montrer le biais.

**Votre panier point-in-time est-il exact ?** Non, c'est une approximation, avec deux limites connues.
Le nombre d'actions bouge dans le temps : Citi a énormément dilué en 2008-2009, donc il paraît trop gros
avant, alors qu'Apple a beaucoup racheté et paraît trop petit. Et les titres disparus de la cote, comme
Celgene ou EMC, n'ont plus de prix sur Yahoo. Mais il reproduit le COR1M presque au point près en moyenne
(28,6 % contre 28,4 %), et il converge vers le panier actuel en 2026.

**Pourquoi SPY et pas le panier comme jambe indice ?** Parce que le CBOE fait comme ça et parce
que l'indice complet est plus diversifié que 50 noms : avec le panier, ρ est mécaniquement
plus haut.

**Vous avez backtesté le vrai livre de dispersion ?** Oui, en variance swaps, roulé chaque mois depuis
2015. Les strikes historiques des titres n'existent pas en données gratuites : on les reconstruit à
partir du VIX et du COR1M (section 4). Résultat : +1,0 point de vol par mois brut, Sharpe 1,18, et
−0,4 point net de coûts (0,5 point de fourchette sur l'indice, 1 point par titre), Sharpe −0,47.
Selon les hypothèses de smile et de coûts, le Sharpe net va de −2,8 à +0,9. Le même livre sur le top 50
d'aujourd'hui affiche 3,6 : encore les survivants.

**Donc la dispersion ne marche pas ?** La prime existe, elle est dans le skew de l'indice. Mais la
capter demande d'acheter 50 variance swaps de titres chaque mois, et la fourchette mange la prime.
C'est exactement ce que disent Driessen, Maenhout et Vilkov : le risque de corrélation est pricé, mais
difficile à capter après coûts. Un desk qui fait de la dispersion gagne sa vie sur l'exécution, le
choix des titres et le timing, pas sur la prime moyenne.

**Le Sharpe du swap est-il crédible ?** Il est brut, sans fourchette (plusieurs points sur un vrai
swap). Sur le panier point-in-time il vaut 0,1 : le chiffre honnête n'a pas besoin d'être défendu,
il dit qu'il n'y a pas de prime à la monnaie.

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

Résultat du contrôle : 40,7 reconstruit contre 41,6 mesuré. Pour le calibrage, on a répliqué la variance
de chaque titre avec la même méthode que le VIX (le strip d'options du vol dashboard). Sur SPY, elle
redonne 17,5 pour un VIX à 17,7.

### 4.7 Les résultats (panier point-in-time)

| Points de variance | 2015-2023 | 2024-2026 |
|---|---|---|
| Prime de l'indice | 70 | 84 |
| dont corrélation | +80 | +56 |
| dont vol individuelle | −10 | +28 |

Trois choses à retenir :

- **Avant 2024, la prime venait de la corrélation.** La part corrélation est positive chaque année depuis
  2007, et 81 % des jours. La vol individuelle, elle, n'est pas surpayée de façon fiable : sa part tourne
  autour de zéro. C'est le résultat de Driessen, Maenhout et Vilkov (Journal of Finance, 2009) : la prime
  de variance de l'indice est une prime de corrélation. On le retrouve avec des données gratuites.
- **La prime est dans le skew, pas à la monnaie.** La corrélation implicite model-free vaut 38 % en
  moyenne depuis 2015, contre 29 % pour le COR1M et 28 % de réalisé. À la monnaie, la corrélation est
  bien pricée (section 1). Ce qui est cher, ce sont les puts de l'indice : ils protègent contre le scénario
  où tous les titres baissent ensemble, c'est-à-dire contre la corrélation qui monte vers 1.
- **Les pertes des pires mois viennent des vols individuelles.** En mars 2020, ce sont les vols des titres
  qui explosent qui font perdre le vendeur de variance indice (70 à 100 % de la perte selon les variantes).
  La part corrélation est la source de revenu régulière.

### 4.8 Ce qui est robuste, et ce qui ne l'est pas

Deux choix pèsent sur les niveaux :
- **Le calibrage.** c_I et c_S ne sont mesurés que sur les jours où l'on a les chaînes. On les traite
  comme des constantes, et on teste trois couples : 0,75/1,05, 0,80/1,10 et 0,85/1,15.
- **Le panier.** Le cas central est le panier point-in-time. Les deux paniers figés servent à mesurer le
  biais : le top 50 d'aujourd'hui gonfle la part corrélation (217 au lieu de 80 avant 2024), le panier des
  31 titres de 2015 manque les titres qui font la variance de l'indice après 2024.

Ce qui tient dans les neuf variantes : la part corrélation est positive avant 2024 ; après 2024 elle
baisse, et la part vol individuelle monte. Ce qui ne tient pas : l'ampleur du déplacement et le signe de
la part individuelle. **On annonce un sens, pas un chiffre.**

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
chaînes du 27 octobre : 40,7 contre 41,6 en moyenne, et une corrélation de 0,89 titre par titre. La
méthode naïve se trompait de 12 %. C'est ce contrôle qui nous a fait corriger la méthode.

**"Pourquoi la prime se déplace vers les titres depuis 2024 ?"** Deux pistes compatibles avec les données
du projet. D'abord, l'indice est devenu quelques géants peu corrélés entre eux : le COR1M est à des plus
bas historiques, donc il reste peu de corrélation à surpayer. Ensuite, ces géants ont des événements
propres très chers, comme les earnings de NVDA ou les annonces IA, qui gonflent leur vol implicite. Ce
sont des hypothèses, pas des résultats démontrés.

**"Quelle est la plus grosse faiblesse ?"** Les deux constantes de calibrage, mesurées sur une seule
journée de chaînes. Le relevé quotidien des chaînes accumule maintenant un historique, ce qui permettra
de les mesurer dans le temps.

## 5. Pitch en trois phrases

"J'ai reconstruit l'indice de corrélation implicite du CBOE depuis 50 chaînes d'options, à 0,6 point
près. J'ai montré que la célèbre prime de corrélation, 13 points par mois sur le top 50 d'aujourd'hui,
était un biais de survivants : sur un panier point-in-time, la corrélation à la monnaie est bien pricée,
et la prime de variance de l'indice se cache dans le skew. Et j'ai backtesté le vrai livre de dispersion
en variance swaps : la prime existe brute, mais 50 fourchettes de prix par mois la mangent, comme le dit la
littérature."
