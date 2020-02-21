import random
import nltk
#nltk.download('words')
from nltk.corpus import words

class CityNames():
    
    def __init__(self):

        seed_cities = []

        # places, people and things related to IBM that would work well in the generator
        seed_cities += [
            ('York','town'),
            ('Rüschli','kon'),
            ('Zür',''),
            ('Ro','chester'),
            ('Johannes','burg'),
            ('Alma','den'),
            ('Böb','lingen'),
            ('Lon','don'),
            ('Burling','ton'),
            ('Ar','monk'),
            ('Dub','lin'),
            ('Wat',''),
            ('Holler','ith'),
            ('Denk',''),
            ('Blue','')
        ]

        # places related to JWO that would work well in the generator
        seed_cities += [
            ('','ly'),
            ('Sut','ton'),
            ('Kirk','stall'),
            ('Bas','el')
        ]

        # places and people related to quantum that would work well in the generator
        seed_cities += [
            ('Aros','a'),
            ('Copen','hagen'),
            ('Cam','bridge'),
            ('Ber','lin'),
            ('Ulm',''),
            ('Bern',''),
            ('Planck',''),
            ('Ein','stein'),
            ('Bohr',''),
            ('Shor',''),
            ('Heisen','berg'),
            ('Pauli',''),
            ('Comp','ton'),
            ('Sommer','feld'),
            ('Grover',''),
            ('Bell','')
        ]
        
        # fantasy sounding placename elements
        seed_cities += [
            ('','helm'),
            ('','rule'),
            ('','fell')
        ]

        firsts = []
        lasts = []
        for city in seed_cities:
            if city[0] not in firsts+['']:
                firsts.append(city[0])
            if city[1] not in lasts+['']:
                lasts.append(city[1])
                
        english = set(words.words())

        self.city_names = []
        for first in firsts:
            for last in lasts:
                if (first[-1]!=last[0] or last[0] in ['t','s']) \
                and ( first+last not in ['sc','tk','ei']) \
                and ( (first,last) not in seed_cities )\
                and ( (first+last).lower() not in english ):
                    self.city_names.append( first+last )
        
        for _ in range(5):
            random.shuffle(self.city_names)
                   
        self._j = 0
        
    def next (self):
        self._j += 1
        try:
            return self.city_names[self._j-1]
        except:
            random.shuffle(self.city_names)
            self._j = 1
            self.city_names[0]
            