'''encoding and decoding for more security inside URLs and session'''
def encode(name):
    '''encodes a string

    using a key and mod 4, the input string is encoded to make it harder to read/recognize/reproduce inside url
    '''
    length = len(name)
    name_encoded = ""
    if length < 4:
        if length == 3:
            return name[1] + name[0] + name[2]
        elif length == 2:
            return name[1] + name[0]
        return name
    key = [2, 0, 1, 3]

    for i in key:
        for j in range(0, length):
            if j % 4 == i:
                name_encoded += name[-j]

    return name_encoded

def decode(name):
    '''decodes a string which was encoded with the method 'encode' above'''
    length = len(name)
    if length < 4:
        if length == 3:
            return name[1] + name[0] + name[2]
        elif length == 2:
            return name[1] + name[0]
        elif length == 1:
            return name
    name_decoded = ""
    rounds = length // 4
    rest = length % 4
    decode_key_base = [2, 4, 1, 3, 2, 4, 1, 3]
    decode_key = []

    for i in range(0, 4):
        decode_key.append(decode_key_base[i+(4-rest)])

    add = 1 if rest > 0 else 0

    for i in range(0, rounds + add):
        for j in decode_key:
            if j == decode_key[0]:
                if i != 0:
                    if rest in [0, 2]:
                        name_decoded += name[j*rounds-i]
                    elif rest == 1:
                        name_decoded += name[j*rounds-i+1]
                    else:
                        name_decoded += name[j*rounds-i+3]
                else:
                    if rest in [0, 1]:
                        name_decoded += name[rounds]
                    elif rest == 2:
                        name_decoded += name[j*rounds-i]
                    else:
                        name_decoded += name[rounds+1]
                if i == rounds and rest < 2:
                    break
            elif j == decode_key[1]:
                if rest in [1, 3]:
                    name_decoded += name[j*rounds-i]
                elif rest == 0:
                    name_decoded += name[j*rounds-i-1]
                else:
                    name_decoded += name[j*rounds-i+1]
                if i == rounds and rest < 3:
                    break
            elif j == decode_key[2]:
                if rest in [1, 2]:
                    name_decoded += name[j*rounds-i]
                elif rest == 0:
                    name_decoded += name[j*rounds-i-1]
                else:
                    name_decoded += name[j*rounds-i+2]
                if i == rounds and rest == 3:
                    break
            else:
                if rest in [0, 1]:
                    name_decoded += name[j*rounds-i-1]
                else:
                    name_decoded += name[j*rounds-i+1]
    return name_decoded
