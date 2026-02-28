## Oppsumering

Et rammeverk for å løse differensiallikninger, som kan f.eks. brukes til å løse dobbel pendel.  

## Hva slags prosjekt er dette?

Utviklingen av prosjektet gikk ut på å lage et rammeverk for å løse differensiallikninger. I begynnelsen var hovedfokuset å løse differensiallikningen "Exponential decay". Dette ble gjort gjennom å implementere en abstrakt klasse ODEModel som er rammeverket for løsning av differensiallikninger. Etter å ha implementert denne, kunne "Exponential decay" bli løst. Med dette rammeverket i ryggen førte dette til at man også kunne løse andre systemet som pendel-systemet og dobbel-pendel-systemet.



## Innhold
Python filer: 
- ode.py
    * Rammeverket for løsning av differensiallikninger.

- exp_decay.py
    * Implementering av differensiallikningen "Exponential decay" som en klasse.

- test_exp_decay.py
    * Testing av exp_dacay.py.

- pendulum.py
    * Implementering av Pendel-systemet som en klasse.

- test_pendulum.py
    * Testing av pendulum.py.

- double_pendulum.py
    * Implementering av dobbelpendel-systemet som en klasse.

- test_double_pendulum.py
    * Testing av double_pendulum.py
## Håndtering

Det er brukt diverse pakker i prosjektet, og det kan ødelegge funkjsonalitet hvis man ikke har disse instalert:
-  numpy
-  scipy
-  dataclasses
-  typing
-  matplotlib
-  abc

Hvis man vil ta i bruk testene, trengs disse pakkene:
-  pytest
-  pathlib

## Simulering

Dersom man kjører double_pendulum.py får man en animasjon av dobbel pendel.

