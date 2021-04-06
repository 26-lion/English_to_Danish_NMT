from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np


encoder_model = load_model("encoder.h5")
decoder_model = load_model("decoder.h5")


eng_vocab = {'is': 1, 'in': 2, 'it': 3, 'during': 4, 'the': 5, 'but': 6, 'and': 7, 'sometimes': 8, 'usually': 9, 'never': 10, 'favorite': 11, 'least': 12, 'fruit': 13, 'most': 14, 'loved': 15, 'liked': 16, 'new': 17, 'paris': 18, 'india': 19, 'united': 20, 'states': 21, 'california': 22, 'jersey': 23, 'france': 24, 'china': 25, 'he': 26, 'she': 27, 'grapefruit': 28, 'your': 29, 'my': 30, 'his': 31, 'her': 32, 'fall': 33, 'june': 34, 'spring': 35, 'january': 36, 'winter': 37, 'march': 38, 'autumn': 39, 'may': 40, 'nice': 41, 'september': 42, 'july': 43, 'april': 44, 'november': 45, 'summer': 46, 'december': 47, 'february': 48, 'our': 49, 'their': 50, 'freezing': 51, 'pleasant': 52, 'beautiful': 53, 'october': 54, 'snowy': 55, 'warm': 56, 'cold': 57, 'wonderful': 58, 'dry': 59, 'busy': 60, 'august': 61, 'chilly': 62, 'rainy': 63, 'mild': 64, 'wet': 65, 'relaxing': 66, 'quiet': 67, 'hot': 68, 'dislikes': 69, 'likes': 70, 'limes': 71, 'lemons': 72, 'grapes': 73, 'mangoes': 74, 'apples': 75, 'peaches': 76, 'oranges': 77, 'pears': 78, 'strawberries': 79, 'bananas': 80, 'to': 81, 'grape': 82, 'apple': 83, 'orange': 84, 'lemon': 85, 'lime': 86, 'banana': 87, 'mango': 88, 'pear': 89, 'strawberry': 90, 'peach': 91, 'like': 92, 'dislike': 93, 'they': 94, 'that': 95, 'i': 96, 'we': 97, 'you': 98, 'animal': 99, 'a': 100, 'truck': 101, 'car': 102, 'automobile': 103, 'was': 104, 'next': 105, 'go': 106, 'driving': 107, 'visit': 108, 'little': 109, 'big': 110, 'old': 111, 'yellow': 112, 'red': 113, 'rusty': 114, 'blue': 115, 'white': 116, 'black': 117, 'green': 118, 'shiny': 119, 'are': 120, 'last': 121, 'feared': 122, 'animals': 123, 'this': 124, 'plan': 125, 'going': 126, 'saw': 127, 'disliked': 128, 'drives': 129, 'drove': 130, 'between': 131, 'translate': 132, 'plans': 133, 'were': 134, 'went': 135, 'might': 136, 'wanted': 137, 'thinks': 138, 'spanish': 139, 'portuguese': 140, 'chinese': 141, 'english': 142, 'french': 143, 'translating': 144, 'difficult': 145, 'fun': 146, 'easy': 147, 'wants': 148, 'think': 149, 'why': 150, "it's": 151, 'did': 152, 'cat': 153, 'shark': 154, 'bird': 155, 'mouse': 156, 'horse': 157, 'elephant': 158, 'dog': 159, 'monkey': 160, 'lion': 161, 'bear': 162, 'rabbit': 163, 'snake': 164, 'when': 165, 'want': 166, 'do': 167, 'how': 168, 'elephants': 169, 'horses': 170, 'dogs': 171, 'sharks': 172, 'snakes': 173, 'cats': 174, 'rabbits': 175, 'monkeys': 176, 'bears': 177, 'birds': 178, 'lions': 179, 'mice': 180, "didn't": 181, 'eiffel': 182, 'tower': 183, 'grocery': 184, 'store': 185, 'football': 186, 'field': 187, 'lake': 188, 'school': 189, 'would': 190, "aren't": 191, 'been': 192, 'weather': 193, 'does': 194, 'has': 195, "isn't": 196, 'am': 197, 'where': 198, 'have': 199}
dan_vocab = {'er': 1, 'start': 2, 'end': 3, 'i': 4, 'det': 5, 'men': 6, 'og': 7, 'om': 8, 'lide': 9, 'normalt': 10, 'aldrig': 11, 'undertiden': 12, 'mindst': 13, 'kan': 14, 'frugt': 15, 'efteråret': 16, 'varmt': 17, 'mest': 18, 'elskede': 19, 'favorit': 20, 'ikke': 21, 'paris': 22, 'indien': 23, 'usa': 24, 'frankrig': 25, 'kina': 26, 'new': 27, 'jersey': 28, 'han': 29, 'hun': 30, 'af': 31, 'californien': 32, 'hans': 33, 'hendes': 34, 'koldt': 35, 'juni': 36, 'januar': 37, 'marts': 38, 'maj': 39, 'september': 40, 'juli': 41, 'april': 42, 'foråret': 43, 'november': 44, 'december': 45, 'februar': 46, 'din': 47, 'min': 48, 'vores': 49, 'deres': 50, 'fryser': 51, 'oktober': 52, 'vinteren': 53, 'yndlingsfrugt': 54, 'sommeren': 55, 'travlt': 56, 'august': 57, 'stille': 58, 'vidunderligt': 59, 'behageligt': 60, 'tørt': 61, 'køligt': 62, 'jordbær': 63, 'regnfuldt': 64, 'vådt': 65, 'slapper': 66, 'smukt': 67, 'mango': 68, 'mildt': 69, 'grapefrugt': 70, 'citroner': 71, 'druer': 72, 'æbler': 73, 'ferskner': 74, 'appelsiner': 75, 'pærer': 76, 'bananer': 77, 'den': 78, 'dejligt': 79, 'limefrugter': 80, 'kalk': 81, 'druen': 82, 'æblet': 83, 'appelsinen': 84, 'citronen': 85, 'bananen': 86, 'pæren': 87, 'sneet': 88, 'grapefrugten': 89, 'foretrukne': 90, 'har': 91, 'godt': 92, 'bil': 93, 'mangoen': 94, 'rart': 95, 'fersken': 96, 'de': 97, 'løbet': 98, 'til': 99, 'jordbæret': 100, 'jeg': 101, 'vi': 102, 'du': 103, 'mild': 104, 'at': 105, 'sne': 106, 'smuk': 107, 'en': 108, 'lastbil': 109, 'afslappende': 110, 'varm': 111, 'næste': 112, 'var': 113, 'dyr': 114, 'yndlingsdyr': 115, 'california': 116, 'snevejr': 117, 'våd': 118, 'kunne': 119, 'kører': 120, 'kørte': 121, 'planlægger': 122, 'besøge': 123, 'fersknen': 124, 'lille': 125, 'skinnende': 126, 'blå': 127, 'oversætte': 128, 'frygtede': 129, 'denne': 130, 'nogle': 131, 'gange': 132, 'sidste': 133, 'ny': 134, 'behagelig': 135, 'store': 136, 'tør': 137, 'så': 138, 'røde': 139, 'gamle': 140, 'gule': 141, 'hvide': 142, 'nye': 143, 'sorte': 144, 'rustne': 145, 'grønne': 146, 'skal': 147, 'dit': 148, 'synes': 149, 'mit': 150, 'snedækket': 151, 'regnfuld': 152, 'mellem': 153, '\u200b\u200bi': 154, 'trøje': 155, 'vil': 156, 'kølig': 157, 'regn': 158, 'tog': 159, 'stor': 160, 'gammel': 161, 'gul': 162, 'grøn': 163, 'hvid': 164, 'sort': 165, 'rød': 166, 'rusten': 167, 'kold': 168, 'spansk': 169, 'portugisisk': 170, 'kinesisk': 171, 'engelsk': 172, 'fransk': 173, 'går': 174, 'måske': 175, 'efterår': 176, 'sjovt': 177, 'let': 178, 'år': 179, 'ville': 180, 'rejse': 181, 'ferskenen': 182, 'vidunderlig': 183, 'lime': 184, 'hvorfor': 185, 'gik': 186, 'svært': 187, 'tage': 188, 'dine': 189, 'mine': 190, 'mus': 191, 'favoritfrugt': 192, 'forår': 193, 'regner': 194, 'sommer': 195, 'vinter': 196, 'hvornår': 197, 'kat': 198, 'haj': 199, 'hest': 200, 'abe': 201, 'bjørn': 202, 'kanin': 203, 'elefant': 204, 'hund': 205, 'slange': 206, 'ønskede': 207, 'fugl': 208, 'løve': 209, 'gå': 210, '\u200b\u200bom': 211, 'under': 212, 'musen': 213, 'muligvis': 214, 'løven': 215, 'hvordan': 216, 'fuglen': 217, 'slangen': 218, 'elefanter': 219, 'heste': 220, 'hunde': 221, 'elefanten': 222, 'hunden': 223, 'hajer': 224, 'aben': 225, 'bjørnen': 226, 'katten': 227, 'kaninen': 228, 'slanger': 229, 'katte': 230, 'kaniner': 231, 'aber': 232, 'bjørne': 233, 'hesten': 234, 'fugle': 235, 'hajen': 236, 'løver': 237, 'eiffeltårnet': 238, 'fodboldbanen': 239, 'søen': 240, 'på': 241, 'købmanden': 242, 'skolen': 243, 'tager': 244, 'gerne': 245, 'fersknet': 246, 'vanskeligt': 247, 'ferskenet': 248, 'små': 249, 'dejlig': 250, 'været': 251, 'besøg': 252, 'vejret': 253, 'hot': 254, 'rejser': 255, 'grapefrugter': 256, 'vanskelig': 257, 'forenede': 258, 'stater': 259, 'regnvejr': 260, 'hvor': 261, 'sneen': 262, 'ønsker': 263, 'march': 264, 'skulle': 265, 'sommetider': 266, 'mindste': 267, 'rejste': 268, 'lidt': 269, 'med': 270, 'købmand': 271, 'skole': 272, 'milde': 273, 'kommer': 274, 'frukt': 275, 'kølige': 276, 'varme': 277, 'komme': 278, 'pæn': 279, 'som': 280, 'regel': 281, 'optaget': 282, 'våde': 283, 'lim': 284, 'rustenblå': 285, 'god': 286}


def padding(sequences, maxLen):
    sequences = pad_sequences(sequences, maxlen=maxLen, padding="post")
    return sequences


def Decode_sequence(seq):
    states_value = encoder_model.predict(seq)
    target_seq = np.zeros((1,1))
    target_seq[0,0] = dan_vocab["start"]
    stop_condition = False
    decoded_sentence = ' '
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)
        sampled_token_index = np.argmax(output_tokens[0, 0, :])
        if sampled_token_index == 0:
            break
        else:
            a = dan_vocab.keys()
            a = list(a)
            sampled_token = a[sampled_token_index-1]
            if sampled_token!='end':
                decoded_sentence += ' '+sampled_token
            if sampled_token == 'end' or len(decoded_sentence.split()) >= 19:
                stop_condition = True
        target_seq = np.zeros((1,1))
        target_seq[0, 0] = sampled_token_index
        states_value = [h, c]
    return decoded_sentence


def encoding(sequence):
    x = []
    warning = " Sorry your text was not in vocabulary, try using names of months, seasons or fruits."
    for i in sequence:
        if i in eng_vocab:
            index = eng_vocab[i]
            x.append(index)
        else:
            return warning
    x = padding([x], 15)
    x = x[0].reshape(1, 15)
    return x

def clean(text):
    punc = '''!()-[]{};:'"\,<>./?@#$%^&*_~1234567890'''
    for elem in text:
        if elem in punc:
            text = text.replace(elem, "")
    return text

app = Flask(__name__)


@app.route('/', methods=["GET", "POST"])
def index():
    if request.method == "POST":
        english = request.form["english"]
        english = english.lower()
        english = clean(english)
        english = list(english.split(" "))
        if len(english) > 15:
            warning2 = "Please input upto 15 words"
            answer=""
            warning=""
        else:
            r = encoding(english)
            if type(r) == str:
                answer = ""
                warning = r
                warning2=""
            else:
                answer = Decode_sequence(r)
                warning = ""
                warning2 = ""
    else:
        answer = ""
        warning = ""
        warning2 = ""
    return render_template("index.html", answer=answer, warning=warning, warning2=warning2)

if __name__ == "__main__":
    app.run(port=8080)
