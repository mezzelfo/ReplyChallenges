# Osservazioni
- Il file '00' si può fare con bruteforce

## Distribuzione dei demoni
### Spazio stamina_consumed vs stamina_recovered vs turns_recovering
- I files '01', '02', '04' sembrano essere caratterizzati dalla stessa distribuzione della generazione dei demoni. In particolare ho trovato con Mathematica le le distribuzioni con cui si generano la quantità di stamina ricoverata e anche il numero di turni per recuperare la stamina:
    
        Round@RandomVariate[CensoredDistribution[{10, 100}, NormalDistribution[49, 20]], 100000];
        KolmogorovSmirnovTest[%, demons[[All, 2]], {"PValue", "ShortTestConclusion"}]

        Round@RandomVariate[CensoredDistribution[{10, 40}, NormalDistribution[7.5, 8.5]], 100000];
        KolmogorovSmirnovTest[%, demons[[All, 3]], {"PValue", "ShortTestConclusion"}]

- Il file '03' è quello contenente più demoni. Nel grafico (X= stamina_consumed, Y = stamina_recovered) si distinguono bene 3 gruppi di demoni, alcuni richiedono poca stamina e ne danno indietro molta (DEMONI BUONI) e altri richiedono molta stamina e ne danno indietro poca (DEMONI CATTIVI). Il terzo gruppo è composto da pochi demino che richiedono poca stamina e ne danno indietro poca (LI POSSIAMO BUTTARE VIA?). Non sembra esserci distizione a livello di turni per recuperare la stamina nei tre gruppi. 
- Il file '05' ha 5 gruppi, uno dei quali sembra ricordare la distribuzione dei files '01','02','04'. Uno ricorda il gruppo piccolo del file '03'. Le differenze rispetto a turns_recovering conviene guardarle su un asse logaritmico
### Spazio vettore frammenti
- Alcuni demoni non danno frammenti. Altri demoni hanno dei turni finali in cui non danno frammenti (si può accorciare la loro sequenza). In particolare circa il 50% dei demoni del file '03' non da frammenti. Il gruppo dei DEMONI BUONI però da SEMPRE 0 frammenti. Servono quindi a ricaricare la stamina
- I files '01' e '02' sembrano comportarsi in modo simile. Il file '04' presenta 3 gruppi di demoni, caratterizzati dall'andamento rispetto al tempo del proprio vettore di frammenti:

        ListPlot@Map[CoefficientList[Fit[#, {1, x}, x], x] &, fragments]

## Osservazioni generali
- Il file 03 forse può essere affrontato parzialmente con la stochastic optimization. Lo stato è (stamina attuale, contatore turno). Le azioni sono due: o affronto un demone buono (e lo stato cambia stocasticamente seguendo la distribuzione dei demoni buoni) o affronto un demone cattivo (e lo stato cambia similmente).