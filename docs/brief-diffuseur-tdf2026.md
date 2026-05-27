# Brief technique — overlay AR temps réel TDF 2026

**À l'attention de** : [contact diffuseur]
**Date** : 2026-05-27
**Objectif** : valider la faisabilité technique et obtenir vos pré-requis
régie avant développement final.

---

## 1. Contexte

Pour le Tour de France 2026 (Grand Départ Florence, 28 juin), nous
proposons un overlay broadcast AR temps réel sur le signal podium
signature équipe : reconstruction 3D du podium en arrière-plan
virtuel + identification automatique des coureurs (nom, équipe,
palmarès) + caméra virtuelle pilotable pour des plans "wow" en
transition broadcast.

## 2. Architecture proposée

```
┌─ PC studio (Linux, RTX 4060) ────────────────────────────┐
│  Capture Magewell SDI (1080p60) → reco automatique :     │
│   • Identification faciale (InsightFace ArcFace,         │
│     index de 782 coureurs validé sur frames TDF 2024)    │
│   • Tracking corps temps réel (YOLOv8-pose + BoT-SORT)   │
│  Sorties : NDI HB vidéo + métadonnées OSC (qui/où)       │
└──────────────────────────────────┬───────────────────────┘
                                   │ LAN 2.5 GbE
                                   ▼
┌─ Box compositor (Windows 11, RTX 3080) ──────────────────┐
│  Unreal Engine 5.4 + plugins NDI / OSC                   │
│   • Scène 3D podium (modèle Blender, ré-éclairé Lumen)   │
│   • Billboards "rider info" suivant chaque coureur       │
│   • Caméra virtuelle pilotable (orbital, zoom)           │
│  Sortie : NDI HD vers votre régie                        │
└──────────────────────────────────────────────────────────┘
```

## 3. Specs techniques garanties

| Item | Cible |
|---|---|
| Résolution | 1080p60 (1080p50 PAL possible si demandé) |
| Latence capture → NDI out | < 100 ms |
| Format NDI | High Bandwidth (SpeedHQ ≈ 150 Mbps) |
| Identification coureurs | Index 782 coureurs, accuracy mesurée sur frames TDF 2024 : à confirmer en test live |
| Failover | 2ème PC compositor en hot standby (à acquérir) ; régie peut basculer sur signal brut en cas de défaillance |
| Stabilité | Tests réels prévus sur épreuves ASO antérieures au TDF (Paris-Nice idéalement) |

## 4. Questions à valider de votre côté

1. **Format d'entrée régie** : votre vision mixer accepte-t-il du NDI HD,
   ou exigez-vous du SDI 3G/12G ? (Si SDI, nous ajoutons un convertisseur
   matériel NDI→SDI en bout de chaîne.)
2. **Latence acceptable** : notre cible <100ms vous va-t-elle, ou
   contraintes plus strictes (e.g. broadcast direct sans délai
   technique) ?
3. **Branding TDF / ASO** : pouvez-vous nous mettre en relation avec
   ASO pour obtenir l'accord d'utilisation des logos officiels
   (TDF, sponsors podium) sur notre scène 3D ? À défaut, nous
   produirons une version anonymisée "esprit TDF" sans logos.
4. **Tests en condition réelle** : possibilité d'un test live sur une
   épreuve ASO antérieure (Paris-Nice mars/avril 2026 idéalement) pour
   valider le workflow avec votre régie ?
5. **Validation workflow** : avez-vous une équipe technique régie qui
   peut auditer notre pipeline en amont (e.g. semaine du 15 juin
   2026), pour s'assurer qu'aucune incompatibilité ne ressorte le
   jour J ?
6. **Specs failover** : exigez-vous un plan de redondance spécifique
   (e.g. SLA hardware, hotline 24/7 pendant l'événement) ?

## 5. Calendrier proposé (5 semaines)

| Semaine | Livrable |
|---|---|
| S1 (27 mai - 2 juin) | Validation brief avec vous + setup UE5 + POC pipeline minimal (asset générique) |
| S2 (3 - 9 juin) | Modélisation 3D podium (Blender) ou photogrammétrie sur site si accès possible |
| S3 (10 - 16 juin) | Intégration billboards rider info + cam virtuelle |
| S4 (17 - 23 juin) | Tests fond vert + tests en condition réelle (idéalement avec votre régie) |
| S5 (24 - 27 juin) | Tuning final + déploiement + tests redondance |
| 28 juin | Grand Départ TDF 2026 — live |

---

**Prochaine étape proposée** : appel téléphonique ou réunion technique
~30 min pour répondre aux questions du § 4 et caler les modalités de
collaboration. Disponibilités cette semaine à votre convenance.

**Contact** : [vos coordonnées]
