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

## 4. Pitch en trois phrases

"J'ai reconstruit l'indice de corrélation implicite du CBOE depuis 50 chaînes d'options, à
0,6 point près. J'ai mesuré le premium de corrélation depuis 2006 et montré qu'il s'est effondré
en 2024 parce que les Mag 7 se sont décorrélés : la part du facteur marché est passée de 45 % à
20 %. Et j'ai montré que vendre de la corrélation, c'est vendre un put sur l'indice, avec la
grille de scénarios qu'un desk regarderait avant de mettre le trade."
