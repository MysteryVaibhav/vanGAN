PyTorch implementation of "Bootstrapping Unsupervised Bilingual Lexicon Induction" (Hauer et al., 2017)

# Setup

```shell
# Install anaconda
# conda create -n bbwe python=3.6 anaconda
conda install pytorch -c pytorch
conda install gensim
pip install editdistance
```


# Extract seed lexicon

```shell
python extract_seed_lexicon.py --src en:data/ukWaC/vocab.txt --trg it:data/itWaC/vocab.txt -o data/lex.en.it.m10000.p100.d25.tsv -v

# 2018-02-23 10:16:13,418/Seed[INFO]: Extracted 229 pairs
# 2018-02-23 10:16:13,418/Seed[INFO]: Write to data/lex.en.it.m10000.p100.d25.tsv
```


# Training

```shell
python main.py --src data/embeddings/en/wacky.txt --trg data/embeddings/it/wacky.txt --lex data/lex.en.it.m10000.p100.d25.tsv -o models/en.it/it25.bin  -v
```

## Example result

The following log shows extracted translation pairs at each iteration.

```shell
2018-02-23 18:57:54,718/Trans[INFO]: Read word embeddings from data/embeddings/en/wacky.txt
2018-02-23 18:59:23,654/Trans[INFO]: Read word embeddings from data/embeddings/it/wacky.txt
2018-02-23 19:00:58,125/Trans[INFO]: Read 229 pairs from data/lex.en.it.m10000.p100.d25.tsv
2018-02-23 19:00:58,128/Trans[INFO]: Seed: 229
2018-02-23 19:01:01,424/Trans[INFO]: [10] loss: 16.83800
2018-02-23 19:01:05,170/Trans[INFO]: [20] loss: 12.61985
2018-02-23 19:01:07,251/Trans[INFO]: [30] loss: 9.84227
2018-02-23 19:01:09,309/Trans[INFO]: [40] loss: 7.94754
2018-02-23 19:01:11,822/Trans[INFO]: [50] loss: 6.60186
2018-02-23 19:01:15,119/Trans[INFO]: [60] loss: 5.60995
2018-02-23 19:01:18,800/Trans[INFO]: [70] loss: 4.85526
2018-02-23 19:01:21,799/Trans[INFO]: [80] loss: 4.26483
2018-02-23 19:01:24,659/Trans[INFO]: [90] loss: 3.79193
2018-02-23 19:01:28,414/Trans[INFO]: [100] loss: 3.40563
2018-02-23 19:01:29,886/Trans[INFO]: [104] loss: 3.27041

2018-02-23 19:11:22,543/Trans[INFO]: june-maggio 0.905
2018-02-23 19:11:22,544/Trans[INFO]: february-maggio 0.884
2018-02-23 19:11:22,544/Trans[INFO]: june-giugno 0.882
2018-02-23 19:11:22,544/Trans[INFO]: january-maggio 0.877
2018-02-23 19:11:22,544/Trans[INFO]: july-maggio 0.869
2018-02-23 19:11:22,544/Trans[INFO]: february-giugno 0.868
2018-02-23 19:11:22,544/Trans[INFO]: october-maggio 0.867
2018-02-23 19:11:22,544/Trans[INFO]: january-giugno 0.862
2018-02-23 19:11:22,544/Trans[INFO]: march-maggio 0.858
2018-02-23 19:11:22,544/Trans[INFO]: june-marzo 0.856
2018-02-23 19:11:22,544/Trans[INFO]: february-marzo 0.854
2018-02-23 19:11:22,544/Trans[INFO]: october-giugno 0.848
2018-02-23 19:11:22,544/Trans[INFO]: july-giugno 0.847
2018-02-23 19:11:22,544/Trans[INFO]: september-maggio 0.846
2018-02-23 19:11:22,544/Trans[INFO]: january-marzo 0.843
2018-02-23 19:11:22,544/Trans[INFO]: march-giugno 0.841
2018-02-23 19:11:22,544/Trans[INFO]: september-giugno 0.832
2018-02-23 19:11:22,544/Trans[INFO]: october-marzo 0.829
2018-02-23 19:11:22,544/Trans[INFO]: june-luglio 0.822
2018-02-23 19:11:22,544/Trans[INFO]: september-marzo 0.820
2018-02-23 19:11:22,544/Trans[INFO]: march-marzo 0.817
2018-02-23 19:11:22,545/Trans[INFO]: july-marzo 0.817
2018-02-23 19:11:22,545/Trans[INFO]: february-luglio 0.816
2018-02-23 19:11:22,545/Trans[INFO]: november-maggio 0.813
2018-02-23 19:11:22,545/Trans[INFO]: january-luglio 0.810
2018-02-23 19:11:25,897/Trans[INFO]: [8] loss: 2.99902

2018-02-23 19:20:58,168/Trans[INFO]: december-dicembre 0.918
2018-02-23 19:20:58,168/Trans[INFO]: august-dicembre 0.891
2018-02-23 19:20:58,169/Trans[INFO]: december-ottobre 0.861
2018-02-23 19:20:58,169/Trans[INFO]: december-gennaio 0.861
2018-02-23 19:20:58,169/Trans[INFO]: december-febbraio 0.860
2018-02-23 19:20:58,169/Trans[INFO]: december-novembre 0.856
2018-02-23 19:20:58,169/Trans[INFO]: august-settembre 0.832
2018-02-23 19:20:58,169/Trans[INFO]: december-settembre 0.831
2018-02-23 19:20:58,169/Trans[INFO]: august-gennaio 0.829
2018-02-23 19:20:58,169/Trans[INFO]: august-febbraio 0.793
2018-02-23 19:20:58,169/Trans[INFO]: august-ottobre 0.791
2018-02-23 19:20:58,169/Trans[INFO]: august-novembre 0.786
2018-02-23 19:20:58,169/Trans[INFO]: december-agosto 0.770
2018-02-23 19:20:58,169/Trans[INFO]: august-agosto 0.733
2018-02-23 19:20:58,169/Trans[INFO]: may-gennaio 0.649
2018-02-23 19:20:58,169/Trans[INFO]: may-settembre 0.619
2018-02-23 19:20:58,169/Trans[INFO]: spring-dicembre 0.612
2018-02-23 19:20:58,169/Trans[INFO]: december-mercoledì 0.595
2018-02-23 19:20:58,169/Trans[INFO]: eight-online 0.595
2018-02-23 19:20:58,169/Trans[INFO]: issues-temi 0.585
2018-02-23 19:20:58,169/Trans[INFO]: six-online 0.576
2018-02-23 19:20:58,169/Trans[INFO]: august-mercoledì 0.573
2018-02-23 19:20:58,169/Trans[INFO]: million-mila 0.572
2018-02-23 19:20:58,169/Trans[INFO]: idea-idea 0.565
2018-02-23 19:20:58,169/Trans[INFO]: thousands-mila 0.565
2018-02-23 19:20:58,715/Trans[INFO]: [2] loss: 3.41774

2018-02-23 19:30:41,817/Trans[INFO]: campaign-campagna 0.696
2018-02-23 19:30:41,818/Trans[INFO]: germany-francia 0.620
2018-02-23 19:30:41,819/Trans[INFO]: africa-francia 0.590
2018-02-23 19:30:41,819/Trans[INFO]: promote-promuovere 0.572
2018-02-23 19:30:41,819/Trans[INFO]: j.-g. 0.559
2018-02-23 19:30:41,819/Trans[INFO]: proposal-intende 0.549
2018-02-23 19:30:41,819/Trans[INFO]: science-scienza 0.544
2018-02-23 19:30:41,819/Trans[INFO]: charles-giuseppe 0.544
2018-02-23 19:30:41,819/Trans[INFO]: text-testi 0.542
2018-02-23 19:30:41,819/Trans[INFO]: president-presidente 0.541
2018-02-23 19:30:41,819/Trans[INFO]: french-francia 0.536
2018-02-23 19:30:41,819/Trans[INFO]: delivered-deliberazione 0.532
2018-02-23 19:30:41,819/Trans[INFO]: charles-antonio 0.528
2018-02-23 19:30:41,819/Trans[INFO]: cultural-culturale 0.520
2018-02-23 19:30:41,819/Trans[INFO]: increase-modificato 0.517
2018-02-23 19:30:41,819/Trans[INFO]: area-zona 0.513
2018-02-23 19:30:41,819/Trans[INFO]: markets-mercato 0.513
2018-02-23 19:30:41,819/Trans[INFO]: america-francia 0.513
2018-02-23 19:30:41,819/Trans[INFO]: relationship-rapporti 0.511
2018-02-23 19:30:41,819/Trans[INFO]: germany-europei 0.510
2018-02-23 19:30:41,819/Trans[INFO]: a.-g. 0.510
2018-02-23 19:30:41,819/Trans[INFO]: richard-stefano 0.509
2018-02-23 19:30:41,819/Trans[INFO]: sport-spettacolo 0.509
2018-02-23 19:30:41,820/Trans[INFO]: dealing-opere 0.508
2018-02-23 19:30:41,820/Trans[INFO]: j.-m. 0.505
2018-02-23 19:30:42,485/Trans[INFO]: [2] loss: 4.37051

2018-02-23 19:35:36,844/Trans[INFO]: commercial-commerciale 0.578
2018-02-23 19:35:36,845/Trans[INFO]: messages-messaggio 0.566
2018-02-23 19:35:36,845/Trans[INFO]: agreement-accordi 0.550
2018-02-23 19:35:36,845/Trans[INFO]: policy-giunta 0.547
2018-02-23 19:35:36,845/Trans[INFO]: policy-ente 0.544
2018-02-23 19:35:36,845/Trans[INFO]: david-mario 0.532
2018-02-23 19:35:36,845/Trans[INFO]: employment-determinato 0.530
2018-02-23 19:35:36,845/Trans[INFO]: tuesday-lunedì 0.522
2018-02-23 19:35:36,845/Trans[INFO]: knowledge-segreteria 0.520
2018-02-23 19:35:36,845/Trans[INFO]: programmes-progetto 0.518
2018-02-23 19:35:36,845/Trans[INFO]: legal-fiscale 0.516
2018-02-23 19:35:36,845/Trans[INFO]: commercial-investimenti 0.515
2018-02-23 19:35:36,845/Trans[INFO]: delivery-n. 0.511
2018-02-23 19:35:36,845/Trans[INFO]: commercial-industriale 0.508
2018-02-23 19:35:36,845/Trans[INFO]: property-conservazione 0.508
2018-02-23 19:35:36,845/Trans[INFO]: medical-medico 0.505
2018-02-23 19:35:36,845/Trans[INFO]: monday-lunedì 0.504
2018-02-23 19:35:36,845/Trans[INFO]: program-magari 0.500
2018-02-23 19:35:36,845/Trans[INFO]: wednesday-sabato 0.499
2018-02-23 19:35:36,846/Trans[INFO]: consultation-regolamento 0.498
2018-02-23 19:35:36,846/Trans[INFO]: wednesday-lunedì 0.497
2018-02-23 19:35:36,846/Trans[INFO]: monday-giovedì 0.496
2018-02-23 19:35:36,846/Trans[INFO]: test-esame 0.496
2018-02-23 19:35:36,846/Trans[INFO]: consultation-n. 0.494
2018-02-23 19:35:36,846/Trans[INFO]: thursday-lunedì 0.493
2018-02-23 19:35:39,511/Trans[INFO]: [10] loss: 4.49357
2018-02-23 19:35:42,276/Trans[INFO]: [20] loss: 3.89809
2018-02-23 19:35:43,225/Trans[INFO]: [23] loss: 3.77941

2018-02-23 19:39:47,298/Trans[INFO]: friday-venerdì 0.794
2018-02-23 19:39:47,299/Trans[INFO]: friday-martedì 0.750
2018-02-23 19:39:47,300/Trans[INFO]: saturday-venerdì 0.727
2018-02-23 19:39:47,300/Trans[INFO]: saturday-martedì 0.675
2018-02-23 19:39:47,300/Trans[INFO]: friday-domenica 0.667
2018-02-23 19:39:47,300/Trans[INFO]: saturday-domenica 0.637
2018-02-23 19:39:47,300/Trans[INFO]: sunday-domenica 0.619
2018-02-23 19:39:47,300/Trans[INFO]: sunday-venerdì 0.612
2018-02-23 19:39:47,300/Trans[INFO]: william-francesco 0.604
2018-02-23 19:39:47,300/Trans[INFO]: sunday-martedì 0.593
2018-02-23 19:39:47,300/Trans[INFO]: m.-c. 0.590
2018-02-23 19:39:47,300/Trans[INFO]: department-ministero 0.587
2018-02-23 19:39:47,300/Trans[INFO]: saturday-domani 0.582
2018-02-23 19:39:47,300/Trans[INFO]: william-andrea 0.580
2018-02-23 19:39:47,300/Trans[INFO]: friday-domani 0.579
2018-02-23 19:39:47,300/Trans[INFO]: information-informazioni 0.561
2018-02-23 19:39:47,300/Trans[INFO]: william-luigi 0.557
2018-02-23 19:39:47,300/Trans[INFO]: james-francesco 0.555
2018-02-23 19:39:47,300/Trans[INFO]: secretary-ministro 0.547
2018-02-23 19:39:47,300/Trans[INFO]: william-a. 0.545
2018-02-23 19:39:47,300/Trans[INFO]: issue-aspetti 0.539
2018-02-23 19:39:47,301/Trans[INFO]: william-giovanni 0.537
2018-02-23 19:39:47,301/Trans[INFO]: department-ministro 0.532
2018-02-23 19:39:47,301/Trans[INFO]: artists-autori 0.528
2018-02-23 19:39:47,301/Trans[INFO]: john-francesco 0.528
2018-02-23 19:39:50,032/Trans[INFO]: [10] loss: 3.92229
2018-02-23 19:39:50,544/Trans[INFO]: [12] loss: 3.84580

2018-02-23 19:43:56,015/Trans[INFO]: seven-web 0.552
2018-02-23 19:43:56,017/Trans[INFO]: page-link 0.547
2018-02-23 19:43:56,017/Trans[INFO]: henry-luca 0.539
2018-02-23 19:43:56,017/Trans[INFO]: chairman-responsabile 0.536
2018-02-23 19:43:56,017/Trans[INFO]: jan-otto 0.536
2018-02-23 19:43:56,018/Trans[INFO]: ed-franco 0.533
2018-02-23 19:43:56,018/Trans[INFO]: henry-angelo 0.530
2018-02-23 19:43:56,018/Trans[INFO]: three-web 0.522
2018-02-23 19:43:56,018/Trans[INFO]: relationships-relazioni 0.520
2018-02-23 19:43:56,018/Trans[INFO]: decisions-decisioni 0.515
2018-02-23 19:43:56,018/Trans[INFO]: list-link 0.512
2018-02-23 19:43:56,018/Trans[INFO]: four-web 0.506
2018-02-23 19:43:56,018/Trans[INFO]: available-utenti 0.505
2018-02-23 19:43:56,018/Trans[INFO]: proposals-deciso 0.503
2018-02-23 19:43:56,018/Trans[INFO]: five-web 0.503
2018-02-23 19:43:56,018/Trans[INFO]: arrangements-atti 0.501
2018-02-23 19:43:56,018/Trans[INFO]: martin-franco 0.500
2018-02-23 19:43:56,018/Trans[INFO]: initiative-iniziativa 0.499
2018-02-23 19:43:56,018/Trans[INFO]: easier-facile 0.498
2018-02-23 19:43:56,018/Trans[INFO]: legislation-club 0.496
2018-02-23 19:43:56,018/Trans[INFO]: like-web 0.493
2018-02-23 19:43:56,018/Trans[INFO]: scientific-scientifico 0.493
2018-02-23 19:43:56,018/Trans[INFO]: event-eventi 0.493
2018-02-23 19:43:56,018/Trans[INFO]: george-giorgio 0.492
2018-02-23 19:43:56,018/Trans[INFO]: relationships-rapporto 0.492
2018-02-23 19:43:56,339/Trans[INFO]: [2] loss: 4.62058

2018-02-23 19:47:59,840/Trans[INFO]: american-spagna 0.568
2018-02-23 19:47:59,842/Trans[INFO]: line-linea 0.552
2018-02-23 19:47:59,842/Trans[INFO]: difficulties-questioni 0.550
2018-02-23 19:47:59,843/Trans[INFO]: car-amico 0.548
2018-02-23 19:47:59,843/Trans[INFO]: ideas-idee 0.534
2018-02-23 19:47:59,843/Trans[INFO]: questions-questioni 0.522
2018-02-23 19:47:59,843/Trans[INFO]: r-l. 0.518
2018-02-23 19:47:59,843/Trans[INFO]: matters-questioni 0.517
2018-02-23 19:47:59,843/Trans[INFO]: growth-crescita 0.515
2018-02-23 19:47:59,843/Trans[INFO]: names-nomi 0.514
2018-02-23 19:47:59,843/Trans[INFO]: construction-opera 0.511
2018-02-23 19:47:59,843/Trans[INFO]: introduction-introduzione 0.510
2018-02-23 19:47:59,843/Trans[INFO]: andrew-carlo 0.507
2018-02-23 19:47:59,843/Trans[INFO]: concerned-importanti 0.504
2018-02-23 19:47:59,843/Trans[INFO]: parties-istituzioni 0.500
2018-02-23 19:47:59,843/Trans[INFO]: requires-disposizioni 0.499
2018-02-23 19:47:59,843/Trans[INFO]: scene-storie 0.497
2018-02-23 19:47:59,843/Trans[INFO]: perfect-appunto 0.497
2018-02-23 19:47:59,843/Trans[INFO]: government-enti 0.495
2018-02-23 19:47:59,843/Trans[INFO]: investment-finanziario 0.490
2018-02-23 19:47:59,843/Trans[INFO]: regulations-norme 0.489
2018-02-23 19:47:59,843/Trans[INFO]: certain-finanziarie 0.488
2018-02-23 19:47:59,843/Trans[INFO]: development-crescita 0.488
2018-02-23 19:47:59,843/Trans[INFO]: religion-morale 0.486
2018-02-23 19:47:59,843/Trans[INFO]: testing-accertamento 0.485
2018-02-23 19:48:00,205/Trans[INFO]: [2] loss: 5.33030

2018-02-23 19:51:55,323/Trans[INFO]: services-azienda 0.520
2018-02-23 19:51:55,324/Trans[INFO]: treatment-prestazioni 0.517
2018-02-23 19:51:55,325/Trans[INFO]: link-collegamento 0.504
2018-02-23 19:51:55,325/Trans[INFO]: next-iniziato 0.503
2018-02-23 19:51:55,325/Trans[INFO]: independent-locale 0.501
2018-02-23 19:51:55,326/Trans[INFO]: answer-dunque 0.500
2018-02-23 19:51:55,326/Trans[INFO]: links-collegamento 0.500
2018-02-23 19:51:55,326/Trans[INFO]: image-immagine 0.497
2018-02-23 19:51:55,326/Trans[INFO]: community-partecipazione 0.495
2018-02-23 19:51:55,326/Trans[INFO]: boat-ciao 0.492
2018-02-23 19:51:55,326/Trans[INFO]: states-parlamento 0.491
2018-02-23 19:51:55,326/Trans[INFO]: every-soltanto 0.488
2018-02-23 19:51:55,326/Trans[INFO]: extensive-realizzazione 0.486
2018-02-23 19:51:55,326/Trans[INFO]: last-sappiamo 0.485
2018-02-23 19:51:55,326/Trans[INFO]: training-comma 0.485
2018-02-23 19:51:55,326/Trans[INFO]: advertising-pubblicità 0.483
2018-02-23 19:51:55,326/Trans[INFO]: arts-musica 0.483
2018-02-23 19:51:55,326/Trans[INFO]: encourage-favorire 0.483
2018-02-23 19:51:55,326/Trans[INFO]: editor-professore 0.482
2018-02-23 19:51:55,326/Trans[INFO]: provision-realizzazione 0.482
2018-02-23 19:51:55,326/Trans[INFO]: regarding-proposto 0.482
2018-02-23 19:51:55,326/Trans[INFO]: emergency-richiesto 0.480
2018-02-23 19:51:55,326/Trans[INFO]: culture-cultura 0.478
2018-02-23 19:51:55,326/Trans[INFO]: professor-dott. 0.477
2018-02-23 19:51:55,326/Trans[INFO]: projects-progetti 0.475
2018-02-23 19:51:58,146/Trans[INFO]: [10] loss: 5.32888
2018-02-23 19:52:00,742/Trans[INFO]: [18] loss: 4.86339

2018-02-23 19:55:47,560/Trans[INFO]: opportunities-iniziative 0.558
2018-02-23 19:55:47,563/Trans[INFO]: paul-pietro 0.556
2018-02-23 19:55:47,563/Trans[INFO]: plans-chiesto 0.534
2018-02-23 19:55:47,563/Trans[INFO]: two-sito 0.526
2018-02-23 19:55:47,563/Trans[INFO]: going-invece 0.525
2018-02-23 19:55:47,563/Trans[INFO]: final-inizia 0.524
2018-02-23 19:55:47,563/Trans[INFO]: day-pomeriggio 0.522
2018-02-23 19:55:47,563/Trans[INFO]: say-dice 0.521
2018-02-23 19:55:47,563/Trans[INFO]: ten-sito 0.520
2018-02-23 19:55:47,563/Trans[INFO]: says-dice 0.509
2018-02-23 19:55:47,563/Trans[INFO]: peter-pietro 0.509
2018-02-23 19:55:47,563/Trans[INFO]: funded-coordinamento 0.505
2018-02-23 19:55:47,563/Trans[INFO]: physical-umana 0.502
2018-02-23 19:55:47,563/Trans[INFO]: ten-on 0.500
2018-02-23 19:55:47,563/Trans[INFO]: relating-relativi 0.500
2018-02-23 19:55:47,563/Trans[INFO]: educational-cooperazione 0.498
2018-02-23 19:55:47,563/Trans[INFO]: brain-ragazza 0.497
2018-02-23 19:55:47,564/Trans[INFO]: relation-relativi 0.495
2018-02-23 19:55:47,564/Trans[INFO]: educational-promozione 0.493
2018-02-23 19:55:47,564/Trans[INFO]: civil-politica 0.492
2018-02-23 19:55:47,564/Trans[INFO]: day-settimana 0.492
2018-02-23 19:55:47,564/Trans[INFO]: peter-marco 0.490
2018-02-23 19:55:47,564/Trans[INFO]: week-settimana 0.488
2018-02-23 19:55:47,564/Trans[INFO]: said-dice 0.488
2018-02-23 19:55:47,564/Trans[INFO]: afternoon-pomeriggio 0.488
2018-02-23 19:55:48,037/Trans[INFO]: [2] loss: 5.37523

2018-02-23 19:59:29,424/Trans[INFO]: offer-clienti 0.530
2018-02-23 19:59:29,425/Trans[INFO]: product-prodotti 0.525
2018-02-23 19:59:29,426/Trans[INFO]: events-manifestazioni 0.511
2018-02-23 19:59:29,427/Trans[INFO]: eu-ue 0.510
2018-02-23 19:59:29,427/Trans[INFO]: countries-nazioni 0.510
2018-02-23 19:59:29,427/Trans[INFO]: economy-economia 0.508
2018-02-23 19:59:29,427/Trans[INFO]: simon-alessandro 0.505
2018-02-23 19:59:29,427/Trans[INFO]: steve-alessandro 0.503
2018-02-23 19:59:29,427/Trans[INFO]: enterprise-economia 0.503
2018-02-23 19:59:29,427/Trans[INFO]: literature-edizione 0.501
2018-02-23 19:59:29,427/Trans[INFO]: communication-comunicazione 0.500
2018-02-23 19:59:29,427/Trans[INFO]: partnership-collaborazione 0.497
2018-02-23 19:59:29,427/Trans[INFO]: britain-italiano 0.492
2018-02-23 19:59:29,427/Trans[INFO]: demand-costo 0.492
2018-02-23 19:59:29,427/Trans[INFO]: ian-alessandro 0.491
2018-02-23 19:59:29,427/Trans[INFO]: details-indicazioni 0.486
2018-02-23 19:59:29,427/Trans[INFO]: system-sistema 0.485
2018-02-23 19:59:29,427/Trans[INFO]: another-qui 0.483
2018-02-23 19:59:29,427/Trans[INFO]: efforts-voglia 0.481
2018-02-23 19:59:29,427/Trans[INFO]: regulation-vigente 0.480
2018-02-23 19:59:29,427/Trans[INFO]: courses-formazione 0.479
2018-02-23 19:59:29,427/Trans[INFO]: attend-clienti 0.479
2018-02-23 19:59:29,427/Trans[INFO]: executive-bilancio 0.479
2018-02-23 19:59:29,427/Trans[INFO]: statutory-violazione 0.479
2018-02-23 19:59:29,428/Trans[INFO]: good-veramente 0.477
2018-02-23 19:59:29,864/Trans[INFO]: [2] loss: 5.95605

2018-02-23 20:05:01,565/Trans[INFO]: history-storia 0.525
2018-02-23 20:05:01,568/Trans[INFO]: sat-sette 0.514
2018-02-23 20:05:01,568/Trans[INFO]: member-membro 0.508
2018-02-23 20:05:01,569/Trans[INFO]: chris-paolo 0.505
2018-02-23 20:05:01,569/Trans[INFO]: debate-discussione 0.498
2018-02-23 20:05:01,569/Trans[INFO]: supply-gestione 0.498
2018-02-23 20:05:01,569/Trans[INFO]: comments-commento 0.495
2018-02-23 20:05:01,569/Trans[INFO]: dr-prof. 0.494
2018-02-23 20:05:01,569/Trans[INFO]: changes-modifica 0.489
2018-02-23 20:05:01,569/Trans[INFO]: mission-missione 0.488
2018-02-23 20:05:01,569/Trans[INFO]: challenges-argomenti 0.488
2018-02-23 20:05:01,569/Trans[INFO]: one-siti 0.483
2018-02-23 20:05:01,570/Trans[INFO]: movement-azione 0.483
2018-02-23 20:05:01,570/Trans[INFO]: please-modulo 0.482
2018-02-23 20:05:01,570/Trans[INFO]: history-letteratura 0.482
2018-02-23 20:05:01,570/Trans[INFO]: hope-comincia 0.482
2018-02-23 20:05:01,570/Trans[INFO]: evening-serata 0.479
2018-02-23 20:05:01,570/Trans[INFO]: change-modifica 0.479
2018-02-23 20:05:01,570/Trans[INFO]: moving-decreto 0.478
2018-02-23 20:05:01,570/Trans[INFO]: necessary-necessaria 0.478
2018-02-23 20:05:01,570/Trans[INFO]: enable-consentire 0.477
2018-02-23 20:05:01,570/Trans[INFO]: really-me 0.477
2018-02-23 20:05:01,570/Trans[INFO]: enable-aggiornamento 0.476
2018-02-23 20:05:01,570/Trans[INFO]: agreed-consente 0.476
2018-02-23 20:05:01,570/Trans[INFO]: thomas-paolo 0.475
2018-02-23 20:05:02,796/Trans[INFO]: [2] loss: 6.44600

2018-02-23 20:12:48,137/Trans[INFO]: applications-applicano 0.506
2018-02-23 20:12:48,140/Trans[INFO]: reform-parlamentare 0.504
2018-02-23 20:12:48,140/Trans[INFO]: field-ambito 0.498
2018-02-23 20:12:48,140/Trans[INFO]: determine-coscienza 0.491
2018-02-23 20:12:48,140/Trans[INFO]: several-software 0.491
2018-02-23 20:12:48,140/Trans[INFO]: god-uomini 0.490
2018-02-23 20:12:48,140/Trans[INFO]: concept-proposta 0.486
2018-02-23 20:12:48,140/Trans[INFO]: market-concorrenza 0.485
2018-02-23 20:12:48,140/Trans[INFO]: meet-copia 0.482
2018-02-23 20:12:48,141/Trans[INFO]: therefore-capacità 0.479
2018-02-23 20:12:48,141/Trans[INFO]: departments-amministrazione 0.479
2018-02-23 20:12:48,141/Trans[INFO]: common-fondamentali 0.479
2018-02-23 20:12:48,141/Trans[INFO]: specifically-specifiche 0.477
2018-02-23 20:12:48,141/Trans[INFO]: st-ragazzi 0.475
2018-02-23 20:12:48,141/Trans[INFO]: field-provincia 0.469
2018-02-23 20:12:48,141/Trans[INFO]: want-adesso 0.469
2018-02-23 20:12:48,141/Trans[INFO]: improve-migliorare 0.468
2018-02-23 20:12:48,141/Trans[INFO]: requirements-finanze 0.468
2018-02-23 20:12:48,141/Trans[INFO]: type-aree 0.465
2018-02-23 20:12:48,141/Trans[INFO]: run-utente 0.459
2018-02-23 20:12:48,141/Trans[INFO]: understanding-organizzazione 0.459
2018-02-23 20:12:48,141/Trans[INFO]: health-sanitario 0.458
2018-02-23 20:12:48,141/Trans[INFO]: assist-attraverso 0.457
2018-02-23 20:12:48,141/Trans[INFO]: several-computer 0.457
2018-02-23 20:12:48,141/Trans[INFO]: modern-tradizione 0.457
2018-02-23 20:12:53,913/Trans[INFO]: [10] loss: 6.25532
2018-02-23 20:12:57,132/Trans[INFO]: [15] loss: 5.93117

2018-02-23 20:20:25,609/Trans[INFO]: able-permette 0.508
2018-02-23 20:20:25,612/Trans[INFO]: researchers-laboratorio 0.508
2018-02-23 20:20:25,613/Trans[INFO]: institute-ufficio 0.505
2018-02-23 20:20:25,613/Trans[INFO]: leader-rappresentanti 0.505
2018-02-23 20:20:25,613/Trans[INFO]: environment-solidarietà 0.501
2018-02-23 20:20:25,613/Trans[INFO]: study-laboratorio 0.500
2018-02-23 20:20:25,613/Trans[INFO]: morning-sera 0.500
2018-02-23 20:20:25,613/Trans[INFO]: studies-analisi 0.499
2018-02-23 20:20:25,613/Trans[INFO]: grant-bis 0.495
2018-02-23 20:20:25,613/Trans[INFO]: staff-responsabili 0.491
2018-02-23 20:20:25,613/Trans[INFO]: broad-recante 0.490
2018-02-23 20:20:25,613/Trans[INFO]: something-qualcosa 0.486
2018-02-23 20:20:25,614/Trans[INFO]: competition-mercati 0.485
2018-02-23 20:20:25,614/Trans[INFO]: grant-finanziamento 0.484
2018-02-23 20:20:25,614/Trans[INFO]: act-ipotesi 0.483
2018-02-23 20:20:25,614/Trans[INFO]: require-attuazione 0.482
2018-02-23 20:20:25,614/Trans[INFO]: organisation-rappresentanza 0.481
2018-02-23 20:20:25,614/Trans[INFO]: technical-docente 0.481
2018-02-23 20:20:25,614/Trans[INFO]: region-quota 0.481
2018-02-23 20:20:25,614/Trans[INFO]: something-insomma 0.479
2018-02-23 20:20:25,614/Trans[INFO]: implementation-funzionamento 0.479
2018-02-23 20:20:25,614/Trans[INFO]: state-governo 0.478
2018-02-23 20:20:25,614/Trans[INFO]: images-immagini 0.478
2018-02-23 20:20:25,614/Trans[INFO]: office-sottosegretario 0.477
2018-02-23 20:20:25,614/Trans[INFO]: place-sede 0.477
2018-02-23 20:20:26,441/Trans[INFO]: [2] loss: 6.37503

2018-02-23 20:27:49,698/Trans[INFO]: indeed-davvero 0.549
2018-02-23 20:27:49,701/Trans[INFO]: invited-invito 0.543
2018-02-23 20:27:49,702/Trans[INFO]: companies-aziende 0.513
2018-02-23 20:27:49,702/Trans[INFO]: service-rete 0.511
2018-02-23 20:27:49,702/Trans[INFO]: business-innovazione 0.508
2018-02-23 20:27:49,702/Trans[INFO]: needed-sufficiente 0.505
2018-02-23 20:27:49,702/Trans[INFO]: human-umanità 0.503
2018-02-23 20:27:49,702/Trans[INFO]: united-internazionale 0.499
2018-02-23 20:27:49,702/Trans[INFO]: need-oneri 0.496
2018-02-23 20:27:49,702/Trans[INFO]: engineering-tecnologia 0.496
2018-02-23 20:27:49,702/Trans[INFO]: online-rete 0.495
2018-02-23 20:27:49,703/Trans[INFO]: laws-leggi 0.492
2018-02-23 20:27:49,703/Trans[INFO]: research-ricerche 0.492
2018-02-23 20:27:49,703/Trans[INFO]: country-paesi 0.490
2018-02-23 20:27:49,703/Trans[INFO]: registration-d.p.r 0.488
2018-02-23 20:27:49,703/Trans[INFO]: allow-eventuali 0.488
2018-02-23 20:27:49,703/Trans[INFO]: sciences-scientifica 0.483
2018-02-23 20:27:49,703/Trans[INFO]: offered-dipendenti 0.478
2018-02-23 20:27:49,703/Trans[INFO]: song-motivi 0.476
2018-02-23 20:27:49,703/Trans[INFO]: sector-agricoltura 0.476
2018-02-23 20:27:49,703/Trans[INFO]: court-imprese 0.476
2018-02-23 20:27:49,703/Trans[INFO]: sell-finanziari 0.474
2018-02-23 20:27:49,703/Trans[INFO]: nice-mai 0.473
2018-02-23 20:27:49,703/Trans[INFO]: suggested-voluto 0.472
2018-02-23 20:27:49,703/Trans[INFO]: offered-soggetti 0.472
2018-02-23 20:27:50,630/Trans[INFO]: [2] loss: 6.77756

2018-02-23 20:35:06,681/Trans[INFO]: establish-finalità 0.534
2018-02-23 20:35:06,683/Trans[INFO]: night-mattina 0.533
2018-02-23 20:35:06,683/Trans[INFO]: trying-bisogna 0.506
2018-02-23 20:35:06,684/Trans[INFO]: activities-manifestazione 0.505
2018-02-23 20:35:06,684/Trans[INFO]: month-giornata 0.493
2018-02-23 20:35:06,684/Trans[INFO]: weekend-mattina 0.492
2018-02-23 20:35:06,684/Trans[INFO]: establish-costituisce 0.491
2018-02-23 20:35:06,685/Trans[INFO]: mr-sindaco 0.488
2018-02-23 20:35:06,685/Trans[INFO]: month-mese 0.488
2018-02-23 20:35:06,685/Trans[INFO]: interactive-progettazione 0.485
2018-02-23 20:35:06,685/Trans[INFO]: established-finalità 0.482
2018-02-23 20:35:06,685/Trans[INFO]: night-estate 0.482
2018-02-23 20:35:06,685/Trans[INFO]: agencies-amministrazioni 0.479
2018-02-23 20:35:06,685/Trans[INFO]: obvious-principali 0.477
2018-02-23 20:35:06,685/Trans[INFO]: education-didattica 0.476
2018-02-23 20:35:06,685/Trans[INFO]: manager-dirigente 0.476
2018-02-23 20:35:06,685/Trans[INFO]: offers-alunni 0.474
2018-02-23 20:35:06,685/Trans[INFO]: cases-operazioni 0.474
2018-02-23 20:35:06,685/Trans[INFO]: course-normativa 0.472
2018-02-23 20:35:06,685/Trans[INFO]: establish-adozione 0.472
2018-02-23 20:35:06,685/Trans[INFO]: facts-verità 0.471
2018-02-23 20:35:06,685/Trans[INFO]: availability-tutela 0.470
2018-02-23 20:35:06,685/Trans[INFO]: organisations-associazioni 0.469
2018-02-23 20:35:06,685/Trans[INFO]: exercise-legge 0.469
2018-02-23 20:35:06,685/Trans[INFO]: selling-consumatori 0.467
2018-02-23 20:35:07,153/Trans[INFO]: [2] loss: 7.17299

2018-02-23 20:42:29,989/Trans[INFO]: commission-consiglio 0.533
2018-02-23 20:42:29,995/Trans[INFO]: strategy-provvedimento 0.522
2018-02-23 20:42:29,997/Trans[INFO]: discussion-votazione 0.519
2018-02-23 20:42:29,997/Trans[INFO]: draft-provvedimento 0.509
2018-02-23 20:42:29,997/Trans[INFO]: written-sai 0.499
2018-02-23 20:42:29,997/Trans[INFO]: discussion-specifico 0.488
2018-02-23 20:42:29,997/Trans[INFO]: individual-ciascuno 0.485
2018-02-23 20:42:29,997/Trans[INFO]: request-serve 0.484
2018-02-23 20:42:29,997/Trans[INFO]: foundation-and 0.484
2018-02-23 20:42:29,998/Trans[INFO]: commission-commissione 0.483
2018-02-23 20:42:29,998/Trans[INFO]: national-nazionale 0.483
2018-02-23 20:42:29,998/Trans[INFO]: systems-sistemi 0.481
2018-02-23 20:42:29,998/Trans[INFO]: involvement-nomina 0.480
2018-02-23 20:42:29,998/Trans[INFO]: meeting-fax 0.480
2018-02-23 20:42:29,998/Trans[INFO]: doctor-padre 0.479
2018-02-23 20:42:29,998/Trans[INFO]: file-file 0.479
2018-02-23 20:42:29,998/Trans[INFO]: similar-specifica 0.479
2018-02-23 20:42:29,998/Trans[INFO]: others-persone 0.476
2018-02-23 20:42:29,998/Trans[INFO]: function-struttura 0.474
2018-02-23 20:42:29,998/Trans[INFO]: commission-nazionale 0.474
2018-02-23 20:42:29,998/Trans[INFO]: loss-ordinanza 0.471
2018-02-23 20:42:29,998/Trans[INFO]: arrived-arriva 0.470
2018-02-23 20:42:29,998/Trans[INFO]: songs-oggetti 0.469
2018-02-23 20:42:29,998/Trans[INFO]: industry-ricerca 0.469
2018-02-23 20:42:29,998/Trans[INFO]: hospital-ospedale 0.469
2018-02-23 20:42:30,470/Trans[INFO]: [2] loss: 7.55152

2018-02-23 20:46:15,643/Trans[INFO]: effective-efficacia 0.514
2018-02-23 20:46:15,646/Trans[INFO]: analysis-verifica 0.499
2018-02-23 20:46:15,647/Trans[INFO]: approved-provvede 0.488
2018-02-23 20:46:15,647/Trans[INFO]: care-sanitaria 0.484
2018-02-23 20:46:15,647/Trans[INFO]: developments-evoluzione 0.481
2018-02-23 20:46:15,647/Trans[INFO]: project-programmi 0.478
2018-02-23 20:46:15,647/Trans[INFO]: workshops-programmi 0.478
2018-02-23 20:46:15,647/Trans[INFO]: welfare-giustizia 0.476
2018-02-23 20:46:15,647/Trans[INFO]: football-calcio 0.473
2018-02-23 20:46:15,647/Trans[INFO]: description-ritiene 0.472
2018-02-23 20:46:15,647/Trans[INFO]: required-necessario 0.471
2018-02-23 20:46:15,647/Trans[INFO]: provided-operatori 0.470
2018-02-23 20:46:15,647/Trans[INFO]: minimum-necessario 0.469
2018-02-23 20:46:15,647/Trans[INFO]: reader-informatica 0.468
2018-02-23 20:46:15,647/Trans[INFO]: town-città 0.467
2018-02-23 20:46:15,647/Trans[INFO]: announced-chiede 0.465
2018-02-23 20:46:15,647/Trans[INFO]: developed-direttiva 0.464
2018-02-23 20:46:15,647/Trans[INFO]: chair-dipartimento 0.464
2018-02-23 20:46:15,647/Trans[INFO]: resource-documentazione 0.463
2018-02-23 20:46:15,647/Trans[INFO]: relevant-programmi 0.462
2018-02-23 20:46:15,647/Trans[INFO]: evaluation-verifica 0.462
2018-02-23 20:46:15,647/Trans[INFO]: click-scheda 0.461
2018-02-23 20:46:15,647/Trans[INFO]: likely-necessari 0.460
2018-02-23 20:46:15,648/Trans[INFO]: essential-evoluzione 0.459
2018-02-23 20:46:15,648/Trans[INFO]: workshops-criteri 0.458
2018-02-23 20:46:16,204/Trans[INFO]: [2] loss: 7.88973

2018-02-23 20:49:09,601/Trans[INFO]: couple-blog 0.534
2018-02-23 20:49:09,601/Trans[INFO]: anyway-niente 0.524
2018-02-23 20:49:09,604/Trans[INFO]: even-anzi 0.519
2018-02-23 20:49:09,605/Trans[INFO]: happen-qualcuno 0.509
2018-02-23 20:49:09,605/Trans[INFO]: procedures-procedure 0.499
2018-02-23 20:49:09,605/Trans[INFO]: couple-pagina 0.498
2018-02-23 20:49:09,605/Trans[INFO]: actually-sicuramente 0.496
2018-02-23 20:49:09,605/Trans[INFO]: maybe-niente 0.494
2018-02-23 20:49:09,605/Trans[INFO]: conference-convenzione 0.490
2018-02-23 20:49:09,605/Trans[INFO]: stories-pagine 0.478
2018-02-23 20:49:09,605/Trans[INFO]: certainly-sicuramente 0.477
2018-02-23 20:49:09,605/Trans[INFO]: ancient-storico 0.476
2018-02-23 20:49:09,605/Trans[INFO]: anyway-qualcuno 0.474
2018-02-23 20:49:09,605/Trans[INFO]: stuff-niente 0.472
2018-02-23 20:49:09,605/Trans[INFO]: exactly-regole 0.469
2018-02-23 20:49:09,605/Trans[INFO]: specific-privati 0.469
2018-02-23 20:49:09,605/Trans[INFO]: blood-spese 0.469
2018-02-23 20:49:09,606/Trans[INFO]: damage-seduta 0.468
2018-02-23 20:49:09,606/Trans[INFO]: specific-singoli 0.468
2018-02-23 20:49:09,606/Trans[INFO]: came-indirizzo 0.463
2018-02-23 20:49:09,606/Trans[INFO]: -comitato 0.463
2018-02-23 20:49:09,606/Trans[INFO]: looked-niente 0.463
2018-02-23 20:49:09,606/Trans[INFO]: placed-sezione 0.461
2018-02-23 20:49:09,606/Trans[INFO]: assessment-prevista 0.461
2018-02-23 20:49:09,606/Trans[INFO]: except-relative 0.461
2018-02-23 20:49:14,026/Trans[INFO]: [10] loss: 7.42793
2018-02-23 20:49:15,056/Trans[INFO]: [12] loss: 7.28041

2018-02-23 20:52:12,067/Trans[INFO]: anything-nessuno 0.519
2018-02-23 20:52:12,070/Trans[INFO]: faith-democrazia 0.515
2018-02-23 20:52:12,070/Trans[INFO]: unable-occorre 0.507
2018-02-23 20:52:12,071/Trans[INFO]: think-gente 0.506
2018-02-23 20:52:12,071/Trans[INFO]: expected-dovuto 0.504
2018-02-23 20:52:12,071/Trans[INFO]: chief-segretario 0.503
2018-02-23 20:52:12,071/Trans[INFO]: st.-giovani 0.503
2018-02-23 20:52:12,071/Trans[INFO]: thing-nessuno 0.499
2018-02-23 20:52:12,071/Trans[INFO]: guidelines-modalità 0.499
2018-02-23 20:52:12,071/Trans[INFO]: process-procedura 0.496
2018-02-23 20:52:12,071/Trans[INFO]: agency-agenzia 0.495
2018-02-23 20:52:12,071/Trans[INFO]: thing-nulla 0.491
2018-02-23 20:52:12,071/Trans[INFO]: planned-dovuto 0.491
2018-02-23 20:52:12,071/Trans[INFO]: join-continuare 0.490
2018-02-23 20:52:12,071/Trans[INFO]: anything-nulla 0.489
2018-02-23 20:52:12,071/Trans[INFO]: whatever-solo 0.486
2018-02-23 20:52:12,071/Trans[INFO]: feel-gente 0.484
2018-02-23 20:52:12,071/Trans[INFO]: ready-basta 0.484
2018-02-23 20:52:12,071/Trans[INFO]: things-nulla 0.482
2018-02-23 20:52:12,071/Trans[INFO]: sure-solo 0.481
2018-02-23 20:52:12,071/Trans[INFO]: ready-chiedere 0.481
2018-02-23 20:52:12,071/Trans[INFO]: expected-basta 0.480
2018-02-23 20:52:12,071/Trans[INFO]: thing-te 0.479
2018-02-23 20:52:12,071/Trans[INFO]: sort-nessuno 0.478
2018-02-23 20:52:12,071/Trans[INFO]: someone-nessuno 0.478
2018-02-23 20:52:12,672/Trans[INFO]: [2] loss: 7.46559

2018-02-23 20:55:04,133/Trans[INFO]: types-strutture 0.533
2018-02-23 20:55:04,136/Trans[INFO]: argument-definito 0.518
2018-02-23 20:55:04,136/Trans[INFO]: flow-percentuale 0.512
2018-02-23 20:55:04,136/Trans[INFO]: obviously-ovviamente 0.512
2018-02-23 20:55:04,137/Trans[INFO]: funds-finanziamenti 0.492
2018-02-23 20:55:04,138/Trans[INFO]: improvement-modifiche 0.492
2018-02-23 20:55:04,138/Trans[INFO]: england-inglese 0.491
2018-02-23 20:55:04,138/Trans[INFO]: go-semplicemente 0.490
2018-02-23 20:55:04,138/Trans[INFO]: develop-sviluppare 0.486
2018-02-23 20:55:04,138/Trans[INFO]: workshop-presentazione 0.484
2018-02-23 20:55:04,138/Trans[INFO]: framework-orientamento 0.484
2018-02-23 20:55:04,138/Trans[INFO]: across-relativo 0.483
2018-02-23 20:55:04,138/Trans[INFO]: combination-strutture 0.481
2018-02-23 20:55:04,138/Trans[INFO]: ask-allora 0.481
2018-02-23 20:55:04,138/Trans[INFO]: british-italia 0.477
2018-02-23 20:55:04,138/Trans[INFO]: agent-stranieri 0.473
2018-02-23 20:55:04,138/Trans[INFO]: go-farlo 0.473
2018-02-23 20:55:04,138/Trans[INFO]: politics-dibattito 0.472
2018-02-23 20:55:04,138/Trans[INFO]: review-articolo 0.472
2018-02-23 20:55:04,138/Trans[INFO]: private-cittadini 0.470
2018-02-23 20:55:04,138/Trans[INFO]: correct-semplicemente 0.469
2018-02-23 20:55:04,138/Trans[INFO]: returned-diventa 0.468
2018-02-23 20:55:04,138/Trans[INFO]: budget-spesa 0.468
2018-02-23 20:55:04,138/Trans[INFO]: operations-impresa 0.467
2018-02-23 20:55:04,138/Trans[INFO]: candidates-organizzazioni 0.467
2018-02-23 20:55:04,834/Trans[INFO]: [2] loss: 7.66112

2018-02-23 20:57:51,778/Trans[INFO]: posted-post 0.526
2018-02-23 20:57:51,780/Trans[INFO]: suggests-afferma 0.523
2018-02-23 20:57:51,781/Trans[INFO]: michael-maria 0.510
2018-02-23 20:57:51,781/Trans[INFO]: external-intero 0.509
2018-02-23 20:57:51,782/Trans[INFO]: institutions-unione 0.500
2018-02-23 20:57:51,782/Trans[INFO]: know-dicono 0.490
2018-02-23 20:57:51,782/Trans[INFO]: support-assistenza 0.490
2018-02-23 20:57:51,782/Trans[INFO]: better-meglio 0.487
2018-02-23 20:57:51,782/Trans[INFO]: topics-contenuti 0.487
2018-02-23 20:57:51,782/Trans[INFO]: city-paese 0.482
2018-02-23 20:57:51,782/Trans[INFO]: simply-altrimenti 0.479
2018-02-23 20:57:51,782/Trans[INFO]: choice-misure 0.478
2018-02-23 20:57:51,782/Trans[INFO]: authorities-autorità 0.475
2018-02-23 20:57:51,783/Trans[INFO]: products-offerta 0.475
2018-02-23 20:57:51,783/Trans[INFO]: i.e.-costituito 0.475
2018-02-23 20:57:51,783/Trans[INFO]: weeks-sentenza 0.473
2018-02-23 20:57:51,783/Trans[INFO]: principles-principi 0.473
2018-02-23 20:57:51,783/Trans[INFO]: fees-altrimenti 0.471
2018-02-23 20:57:51,783/Trans[INFO]: investigation-indagini 0.471
2018-02-23 20:57:51,783/Trans[INFO]: challenge-ragione 0.469
2018-02-23 20:57:51,783/Trans[INFO]: products-servizi 0.468
2018-02-23 20:57:51,783/Trans[INFO]: public-popoli 0.465
2018-02-23 20:57:51,783/Trans[INFO]: society-guerra 0.464
2018-02-23 20:57:51,783/Trans[INFO]: assistance-autonomia 0.464
2018-02-23 20:57:51,783/Trans[INFO]: digital-digitale 0.463
2018-02-23 20:57:52,477/Trans[INFO]: [2] loss: 7.88378
100%|█████████████████████████████████████████████████| 1560/1560 [02:37<00:00,  9.90it/s]
2018-02-23 21:00:30,206/Trans[INFO]: changing-modificazioni 0.530
2018-02-23 21:00:30,208/Trans[INFO]: visual-cinema 0.530
2018-02-23 21:00:30,209/Trans[INFO]: unless-eventualmente 0.497
2018-02-23 21:00:30,210/Trans[INFO]: order-necessità 0.495
2018-02-23 21:00:30,210/Trans[INFO]: interested-migliori 0.490
2018-02-23 21:00:30,210/Trans[INFO]: funding-fondi 0.488
2018-02-23 21:00:30,210/Trans[INFO]: consideration-pertanto 0.484
2018-02-23 21:00:30,210/Trans[INFO]: interview-commi 0.480
2018-02-23 21:00:30,210/Trans[INFO]: attempt-riesce 0.473
2018-02-23 21:00:30,211/Trans[INFO]: statement-dichiarazioni 0.472
2018-02-23 21:00:30,211/Trans[INFO]: proper-essa 0.472
2018-02-23 21:00:30,211/Trans[INFO]: allows-altresì 0.469
2018-02-23 21:00:30,211/Trans[INFO]: camera-the 0.468
2018-02-23 21:00:30,211/Trans[INFO]: told-spiega 0.467
2018-02-23 21:00:30,211/Trans[INFO]: saying-pensa 0.465
2018-02-23 21:00:30,211/Trans[INFO]: canada-parigi 0.464
2018-02-23 21:00:30,211/Trans[INFO]: purchase-pagamento 0.464
2018-02-23 21:00:30,211/Trans[INFO]: action-richiesta 0.463
2018-02-23 21:00:30,211/Trans[INFO]: explain-sottolinea 0.462
2018-02-23 21:00:30,211/Trans[INFO]: supplied-impianti 0.462
2018-02-23 21:00:30,211/Trans[INFO]: proper-tuttavia 0.462
2018-02-23 21:00:30,212/Trans[INFO]: enough-so 0.461
2018-02-23 21:00:30,212/Trans[INFO]: form-elenco 0.460
2018-02-23 21:00:30,212/Trans[INFO]: plus-casi 0.460
2018-02-23 21:00:30,212/Trans[INFO]: communities-comune 0.459
2018-02-23 21:00:30,889/Trans[INFO]: [2] loss: 8.07914
1
2018-02-23 21:02:55,924/Trans[INFO]: bad-lì 0.531
2018-02-23 21:02:55,926/Trans[INFO]: photo-foto 0.509
2018-02-23 21:02:55,927/Trans[INFO]: universities-banche 0.499
2018-02-23 21:02:55,927/Trans[INFO]: ensure-assicurare 0.490
2018-02-23 21:02:55,928/Trans[INFO]: deal-contratto 0.487
2018-02-23 21:02:55,928/Trans[INFO]: session-incontri 0.483
2018-02-23 21:02:55,928/Trans[INFO]: try-dovrebbero 0.482
2018-02-23 21:02:55,928/Trans[INFO]: learning-competenza 0.479
2018-02-23 21:02:55,928/Trans[INFO]: winter-notte 0.466
2018-02-23 21:02:55,928/Trans[INFO]: nothing-forse 0.463
2018-02-23 21:02:55,928/Trans[INFO]: popular-soprattutto 0.462
2018-02-23 21:02:55,928/Trans[INFO]: knows-detto 0.460
2018-02-23 21:02:55,928/Trans[INFO]: welcome-vedo 0.459
2018-02-23 21:02:55,928/Trans[INFO]: pictures-foto 0.459
2018-02-23 21:02:55,928/Trans[INFO]: bad-probabilmente 0.459
2018-02-23 21:02:55,928/Trans[INFO]: give-peraltro 0.459
2018-02-23 21:02:55,928/Trans[INFO]: applied-nonché 0.458
2018-02-23 21:02:55,928/Trans[INFO]: wait-no 0.458
2018-02-23 21:02:55,928/Trans[INFO]: waiting-detto 0.456
2018-02-23 21:02:55,928/Trans[INFO]: meant-vuole 0.456
2018-02-23 21:02:55,928/Trans[INFO]: meant-significa 0.454
2018-02-23 21:02:55,928/Trans[INFO]: ways-proposte 0.452
2018-02-23 21:02:55,929/Trans[INFO]: advice-spalle 0.452
2018-02-23 21:02:55,929/Trans[INFO]: journey-stagione 0.452
2018-02-23 21:02:55,929/Trans[INFO]: learning-relativa 0.451
2018-02-23 21:02:56,584/Trans[INFO]: [2] loss: 8.23343

2018-02-23 21:05:21,367/Trans[INFO]: allowing-eventuale 0.531
2018-02-23 21:05:21,369/Trans[INFO]: historical-storica 0.501
2018-02-23 21:05:21,370/Trans[INFO]: agenda-statuto 0.499
2018-02-23 21:05:21,371/Trans[INFO]: central-istituzionale 0.499
2018-02-23 21:05:21,372/Trans[INFO]: response-stabilito 0.491
2018-02-23 21:05:21,372/Trans[INFO]: e-s. 0.488
2018-02-23 21:05:21,372/Trans[INFO]: topic-argomento 0.483
2018-02-23 21:05:21,372/Trans[INFO]: advance-uffici 0.481
2018-02-23 21:05:21,372/Trans[INFO]: different-categorie 0.477
2018-02-23 21:05:21,372/Trans[INFO]: improving-miglioramento 0.475
2018-02-23 21:05:21,373/Trans[INFO]: clients-professionali 0.472
2018-02-23 21:05:21,373/Trans[INFO]: aid-sostegno 0.469
2018-02-23 21:05:21,373/Trans[INFO]: financial-economici 0.468
2018-02-23 21:05:21,373/Trans[INFO]: help-bisogno 0.466
2018-02-23 21:05:21,373/Trans[INFO]: phase-assenza 0.465
2018-02-23 21:05:21,373/Trans[INFO]: group-territoriale 0.463
2018-02-23 21:05:21,373/Trans[INFO]: head-rappresentante 0.456
2018-02-23 21:05:21,373/Trans[INFO]: already-materiale 0.455
2018-02-23 21:05:21,373/Trans[INFO]: railway-tv 0.455
2018-02-23 21:05:21,373/Trans[INFO]: involved-azioni 0.454
2018-02-23 21:05:21,373/Trans[INFO]: amount-importo 0.454
2018-02-23 21:05:21,373/Trans[INFO]: due-approvato 0.454
2018-02-23 21:05:21,374/Trans[INFO]: thought-bisogno 0.453
2018-02-23 21:05:21,374/Trans[INFO]: continuing-dovrà 0.453
2018-02-23 21:05:21,374/Trans[INFO]: factors-fattori 0.452
2018-02-23 21:05:21,993/Trans[INFO]: [2] loss: 8.41232

2018-02-23 21:07:50,789/Trans[INFO]: various-pc 0.535
2018-02-23 21:07:50,792/Trans[INFO]: related-ambientali 0.516
2018-02-23 21:07:50,793/Trans[INFO]: sense-espressione 0.506
2018-02-23 21:07:50,795/Trans[INFO]: heritage-culturali 0.498
2018-02-23 21:07:50,795/Trans[INFO]: businesses-commerciali 0.491
2018-02-23 21:07:50,795/Trans[INFO]: basic-fondamentale 0.490
2018-02-23 21:07:50,795/Trans[INFO]: contemporary-poesia 0.489
2018-02-23 21:07:50,795/Trans[INFO]: fields-attualmente 0.488
2018-02-23 21:07:50,795/Trans[INFO]: related-individuazione 0.488
2018-02-23 21:07:50,795/Trans[INFO]: way-facilmente 0.481
2018-02-23 21:07:50,795/Trans[INFO]: defined-indicati 0.479
2018-02-23 21:07:50,795/Trans[INFO]: related-derivanti 0.474
2018-02-23 21:07:50,795/Trans[INFO]: teaching-docenti 0.466
2018-02-23 21:07:50,795/Trans[INFO]: use-esecuzione 0.466
2018-02-23 21:07:50,795/Trans[INFO]: also-effettivamente 0.462
2018-02-23 21:07:50,795/Trans[INFO]: met-mandato 0.461
2018-02-23 21:07:50,796/Trans[INFO]: terms-quadro 0.460
2018-02-23 21:07:50,796/Trans[INFO]: believe-certamente 0.460
2018-02-23 21:07:50,796/Trans[INFO]: spread-incremento 0.458
2018-02-23 21:07:50,796/Trans[INFO]: strategic-consigliere 0.458
2018-02-23 21:07:50,796/Trans[INFO]: ever-effettivamente 0.457
2018-02-23 21:07:50,796/Trans[INFO]: winning-titoli 0.456
2018-02-23 21:07:50,796/Trans[INFO]: london-fondazione 0.455
2018-02-23 21:07:50,796/Trans[INFO]: specialist-titolare 0.453
2018-02-23 21:07:50,796/Trans[INFO]: software-valori 0.451
2018-02-23 21:07:51,473/Trans[INFO]: [2] loss: 8.57209

2018-02-23 21:10:16,068/Trans[INFO]: allowed-obbligo 0.494
2018-02-23 21:10:16,069/Trans[INFO]: caused-presentato 0.472
2018-02-23 21:10:16,071/Trans[INFO]: senior-competente 0.469
2018-02-23 21:10:16,072/Trans[INFO]: sessions-processi 0.459
2018-02-23 21:10:16,073/Trans[INFO]: important-condizione 0.456
2018-02-23 21:10:16,074/Trans[INFO]: tried-cerca 0.455
2018-02-23 21:10:16,074/Trans[INFO]: policies-privato 0.455
2018-02-23 21:10:16,074/Trans[INFO]: secondary-visita 0.455
2018-02-23 21:10:16,074/Trans[INFO]: appropriate-opportuno 0.455
2018-02-23 21:10:16,074/Trans[INFO]: context-svolgimento 0.454
2018-02-23 21:10:16,074/Trans[INFO]: senior-iscrizione 0.450
2018-02-23 21:10:16,074/Trans[INFO]: famous-tanti 0.450
2018-02-23 21:10:16,074/Trans[INFO]: owner-cittadino 0.450
2018-02-23 21:10:16,074/Trans[INFO]: recommendations-scelte 0.450
2018-02-23 21:10:16,074/Trans[INFO]: always-certo 0.449
2018-02-23 21:10:16,074/Trans[INFO]: planning-scelte 0.448
2018-02-23 21:10:16,074/Trans[INFO]: approach-forme 0.446
2018-02-23 21:10:16,074/Trans[INFO]: period-durata 0.446
2018-02-23 21:10:16,074/Trans[INFO]: law-università 0.444
2018-02-23 21:10:16,074/Trans[INFO]: maintenance-funzione 0.443
2018-02-23 21:10:16,074/Trans[INFO]: debt-lavoro 0.442
2018-02-23 21:10:16,074/Trans[INFO]: comprehensive-diffusione 0.442
2018-02-23 21:10:16,074/Trans[INFO]: rules-generali 0.442
2018-02-23 21:10:16,074/Trans[INFO]: whether-cioè 0.441
2018-02-23 21:10:16,074/Trans[INFO]: students-insegnanti 0.441
2018-02-23 21:10:16,747/Trans[INFO]: [2] loss: 8.75066

2018-02-23 21:12:35,584/Trans[INFO]: justice-riforma 0.512
2018-02-23 21:12:35,587/Trans[INFO]: affairs-presidenza 0.507
2018-02-23 21:12:35,588/Trans[INFO]: work-lavori 0.502
2018-02-23 21:12:35,590/Trans[INFO]: probably-nemmeno 0.496
2018-02-23 21:12:35,590/Trans[INFO]: year-giorno 0.483
2018-02-23 21:12:35,590/Trans[INFO]: move-autorizzazione 0.483
2018-02-23 21:12:35,590/Trans[INFO]: us-italiane 0.481
2018-02-23 21:12:35,590/Trans[INFO]: fact-nemmeno 0.480
2018-02-23 21:12:35,590/Trans[INFO]: teams-corsi 0.480
2018-02-23 21:12:35,590/Trans[INFO]: industrial-produzione 0.479
2018-02-23 21:12:35,590/Trans[INFO]: classes-dimensione 0.479
2018-02-23 21:12:35,590/Trans[INFO]: guidance-emendamenti 0.476
2018-02-23 21:12:35,590/Trans[INFO]: programs-soluzioni 0.467
2018-02-23 21:12:35,590/Trans[INFO]: seem-nemmeno 0.466
2018-02-23 21:12:35,590/Trans[INFO]: green-verdi 0.465
2018-02-23 21:12:35,590/Trans[INFO]: child-donna 0.464
2018-02-23 21:12:35,590/Trans[INFO]: everyone-nessun 0.463
2018-02-23 21:12:35,590/Trans[INFO]: guidance-osservazioni 0.463
2018-02-23 21:12:35,590/Trans[INFO]: games-reti 0.462
2018-02-23 21:12:35,590/Trans[INFO]: commitment-approvazione 0.462
2018-02-23 21:12:35,590/Trans[INFO]: communications-informazione 0.459
2018-02-23 21:12:35,590/Trans[INFO]: facilities-biblioteca 0.459
2018-02-23 21:12:35,590/Trans[INFO]: library-biblioteca 0.458
2018-02-23 21:12:35,590/Trans[INFO]: reference-indicazione 0.457
2018-02-23 21:12:35,590/Trans[INFO]: everything-neanche 0.457
2018-02-23 21:12:36,317/Trans[INFO]: [2] loss: 8.91286

2018-02-23 21:16:03,665/Trans[INFO]: create-creare 0.523
2018-02-23 21:16:03,668/Trans[INFO]: rights-diritti 0.521
2018-02-23 21:16:03,669/Trans[INFO]: rates-condizioni 0.521
2018-02-23 21:16:03,670/Trans[INFO]: apply-sensi 0.501
2018-02-23 21:16:03,670/Trans[INFO]: venue-comunale 0.489
2018-02-23 21:16:03,670/Trans[INFO]: get-perchè 0.480
2018-02-23 21:16:03,670/Trans[INFO]: summer-natale 0.476
2018-02-23 21:16:03,670/Trans[INFO]: rates-risorse 0.476
2018-02-23 21:16:03,670/Trans[INFO]: job-moglie 0.474
2018-02-23 21:16:03,670/Trans[INFO]: distribution-distribuzione 0.469
2018-02-23 21:16:03,670/Trans[INFO]: happy-piace 0.469
2018-02-23 21:16:03,670/Trans[INFO]: uk-membri 0.467
2018-02-23 21:16:03,670/Trans[INFO]: tell-giovane 0.464
2018-02-23 21:16:03,670/Trans[INFO]: never-finalmente 0.463
2018-02-23 21:16:03,670/Trans[INFO]: manage-risorse 0.463
2018-02-23 21:16:03,670/Trans[INFO]: multiple-aziendale 0.462
2018-02-23 21:16:03,670/Trans[INFO]: pdf-allegato 0.456
2018-02-23 21:16:03,670/Trans[INFO]: straight-finalmente 0.456
2018-02-23 21:16:03,670/Trans[INFO]: provide-garantire 0.456
2018-02-23 21:16:03,670/Trans[INFO]: wanted-piace 0.455
2018-02-23 21:16:03,671/Trans[INFO]: aim-scopo 0.455
2018-02-23 21:16:03,671/Trans[INFO]: distance-relazione 0.454
2018-02-23 21:16:03,671/Trans[INFO]: integrated-organi 0.454
2018-02-23 21:16:03,671/Trans[INFO]: near-interessati 0.454
2018-02-23 21:16:03,671/Trans[INFO]: rule-regime 0.454
2018-02-23 21:16:05,134/Trans[INFO]: [2] loss: 9.05708

2018-02-23 21:20:22,268/Trans[INFO]: appointed-provvedimenti 0.514
2018-02-23 21:20:22,271/Trans[INFO]: recruitment-provvedimenti 0.511
2018-02-23 21:20:22,272/Trans[INFO]: costs-ragioni 0.497
2018-02-23 21:20:22,273/Trans[INFO]: strategies-piani 0.481
2018-02-23 21:20:22,274/Trans[INFO]: strategies-strade 0.479
2018-02-23 21:20:22,274/Trans[INFO]: wrong-capito 0.465
2018-02-23 21:20:22,274/Trans[INFO]: report-notizie 0.464
2018-02-23 21:20:22,274/Trans[INFO]: student-ragazzo 0.458
2018-02-23 21:20:22,274/Trans[INFO]: thank-dollari 0.458
2018-02-23 21:20:22,274/Trans[INFO]: division-istituti 0.452
2018-02-23 21:20:22,274/Trans[INFO]: paragraph-provvedimenti 0.451
2018-02-23 21:20:22,274/Trans[INFO]: kind-naturalmente 0.451
2018-02-23 21:20:22,274/Trans[INFO]: reports-notizie 0.450
2018-02-23 21:20:22,274/Trans[INFO]: talking-pure 0.449
2018-02-23 21:20:22,274/Trans[INFO]: managing-candidato 0.449
2018-02-23 21:20:22,274/Trans[INFO]: success-libertà 0.446
2018-02-23 21:20:22,274/Trans[INFO]: university-diritto 0.445
2018-02-23 21:20:22,275/Trans[INFO]: kind-tanto 0.444
2018-02-23 21:20:22,275/Trans[INFO]: south-partiti 0.444
2018-02-23 21:20:22,275/Trans[INFO]: let-dichiara 0.443
2018-02-23 21:20:22,275/Trans[INFO]: fantastic-protagonista 0.443
2018-02-23 21:20:22,275/Trans[INFO]: takes-disposto 0.442
2018-02-23 21:20:22,275/Trans[INFO]: charges-ragioni 0.442
2018-02-23 21:20:22,275/Trans[INFO]: method-prospettiva 0.442
2018-02-23 21:20:22,275/Trans[INFO]: knew-sa 0.440
2018-02-23 21:20:23,360/Trans[INFO]: [2] loss: 9.22915

2018-02-23 21:24:56,715/Trans[INFO]: foreign-esteri 0.541
2018-02-23 21:24:56,719/Trans[INFO]: currently- 0.523
2018-02-23 21:24:56,719/Trans[INFO]: people-famiglie 0.518
2018-02-23 21:24:56,720/Trans[INFO]: district-popolazione 0.495
2018-02-23 21:24:56,720/Trans[INFO]: interesting-interessante 0.490
2018-02-23 21:24:56,720/Trans[INFO]: mr.-onorevole 0.488
2018-02-23 21:24:56,720/Trans[INFO]: mr.-commissario 0.481
2018-02-23 21:24:56,720/Trans[INFO]: friend-amici 0.474
2018-02-23 21:24:56,720/Trans[INFO]: advanced-valutazione 0.472
2018-02-23 21:24:56,720/Trans[INFO]: access-inserimento 0.471
2018-02-23 21:24:56,720/Trans[INFO]: video-video 0.471
2018-02-23 21:24:56,720/Trans[INFO]: taught-valutazione 0.469
2018-02-23 21:24:56,720/Trans[INFO]: word-testo 0.469
2018-02-23 21:24:56,720/Trans[INFO]: using-determinazione 0.468
2018-02-23 21:24:56,720/Trans[INFO]: taught-esperienza 0.467
2018-02-23 21:24:56,720/Trans[INFO]: provides-inserimento 0.465
2018-02-23 21:24:56,720/Trans[INFO]: l-b 0.462
2018-02-23 21:24:56,720/Trans[INFO]: english-of 0.460
2018-02-23 21:24:56,720/Trans[INFO]: experiences-esperienze 0.458
2018-02-23 21:24:56,720/Trans[INFO]: taught-legislativo 0.457
2018-02-23 21:24:56,721/Trans[INFO]: suitable-efficace 0.454
2018-02-23 21:24:56,721/Trans[INFO]: particular-nazionali 0.453
2018-02-23 21:24:56,721/Trans[INFO]: meetings-assemblea 0.453
2018-02-23 21:24:56,721/Trans[INFO]: suitable-qualora 0.452
2018-02-23 21:24:56,721/Trans[INFO]: busy-neppure 0.450
```


# Evaluation

```shell
python eval.py data/eval/OPUS.en.it.europarl.txt --src ~/devel/bootstrapping-bwe/data/embeddings/en/wacky.txt --trg ~/devel/bootstrapping-bwe/data/embeddings/it/wacky.txt -m models/it25.bin -k 1 -v
# 0.014 (OOV=0)
python eval.py data/eval/OPUS.en.it.europarl.txt --src ~/devel/bootstrapping-bwe/data/embeddings/en/wacky.txt --trg ~/devel/bootstrapping-bwe/data/embeddings/it/wacky.txt -m models/it25.bin -k 5 -v
# 0.00746666666667 (OOV=0)


python eval.py data/eval/MUSE.en.it.5000-6500.txt --src ~/devel/bootstrapping-bwe/data/embeddings/en/wacky.txt --trg ~/devel/bootstrapping-bwe/data/embeddings/it/wacky.txt -m models/it25.bin -k 1 -v
# 0.0113107119095 (OOV=3)
python eval.py data/eval/MUSE.en.it.5000-6500.txt --src ~/devel/bootstrapping-bwe/data/embeddings/en/wacky.txt --trg ~/devel/bootstrapping-bwe/data/embeddings/it/wacky.txt -m models/it25.bin -k 5 -v
# 0.00665335994677 (OOV=3)
```

