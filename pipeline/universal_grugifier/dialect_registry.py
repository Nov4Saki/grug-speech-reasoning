"""
Top 50 Languages & Regional Dialect Registry for Universal Grugifier
Provides dialect metadata, cultural linguistic markers, and idioms.
"""

from typing import Dict, List

TOP_50_LANGUAGES_AND_DIALECTS: Dict[str, Dict[str, str]] = {
    # 1-10: Major Global Languages & Key Dialects
    "en_aave": {
        "lang": "English",
        "dialect": "African American Vernacular English (AAVE)",
        "region": "North America",
        "markers": "habitual be, copula deletion, lexical markers (finna, boutta)",
        "sample_greeting": "What it do, fam?"
    },
    "en_cockney": {
        "lang": "English",
        "dialect": "Cockney / East London",
        "region": "United Kingdom",
        "markers": "rhyming slang, glottal stop, 'innit', 'proper sorted'",
        "sample_greeting": "Right then, guv'nor, what's all this about?"
    },
    "en_scottish": {
        "lang": "English",
        "dialect": "Scottish English",
        "region": "Scotland",
        "markers": "ken, bonnie, wee, braw, dinnae",
        "sample_greeting": "Aye, how's it gaun, pal?"
    },
    "es_mexican": {
        "lang": "Spanish",
        "dialect": "Mexican Spanish",
        "region": "Mexico",
        "markers": "que onda, neta, orita, chido, chale",
        "sample_greeting": "Que onda carnal, que traes en mente?"
    },
    "es_argentine": {
        "lang": "Spanish",
        "dialect": "Rioplatense (Argentine)",
        "region": "Argentina / Uruguay",
        "markers": "voseo (vos tenes), che, boludo, re-copado",
        "sample_greeting": "Che, como andas? Decime que necesitas resolver."
    },
    "es_castilian": {
        "lang": "Spanish",
        "dialect": "Castilian Spanish",
        "region": "Spain",
        "markers": "distincion, vosotros, guay, molar, vale",
        "sample_greeting": "Hola, que tal estais? Vamos a verlo."
    },
    "ar_egyptian": {
        "lang": "Arabic",
        "dialect": "Egyptian Masri",
        "region": "Egypt",
        "markers": "ezzayak, keda, bardo, da/di, ya basha",
        "sample_greeting": "ازيك يا باشا، ايه الحكاية النهاردة؟"
    },
    "ar_levantine": {
        "lang": "Arabic",
        "dialect": "Shami (Levantine)",
        "region": "Lebanon / Syria / Jordan",
        "markers": "kifak, shu fi ma fi, halaq, emta, ya zalameh",
        "sample_greeting": "كيفك يا غالي، شو القصة بالزبط؟"
    },
    "ar_gulf": {
        "lang": "Arabic",
        "dialect": "Khaleeji (Gulf)",
        "region": "Saudi Arabia / UAE / Kuwait",
        "markers": "shlonak, zain, wallah, ya teir, shuf",
        "sample_greeting": "شلونك عساك طيب، شنو المشكلة عندك؟"
    },
    "zh_mandarin": {
        "lang": "Chinese",
        "dialect": "Standard Mandarin",
        "region": "China / Taiwan / Singapore",
        "markers": "erhua, standard syntax, zhege, nage",
        "sample_greeting": "你好，请问这个问题我们该如何处理？"
    },
    "zh_cantonese": {
        "lang": "Chinese",
        "dialect": "Cantonese (Yue)",
        "region": "Guangdong / Hong Kong",
        "markers": "dim a, mou man tai, ge, laa, zung",
        "sample_greeting": "點呀大佬，呢個問題點樣搞好？"
    },
    "zh_sichuanese": {
        "lang": "Chinese",
        "dialect": "Sichuanese (Southwestern Mandarin)",
        "region": "Sichuan / Chongqing",
        "markers": "xiongdi, ba shi, ga ma ti, sao pi",
        "sample_greeting": "兄弟伙，这个事情巴适得很，我来给你盘。"
    },
    "hi_standard": {
        "lang": "Hindi",
        "dialect": "Khariboli / Standard Hindi",
        "region": "India",
        "markers": "namaste, aap, kya hal hai, samasya",
        "sample_greeting": "नमस्ते, इस गणना को कैसे सुलझाना है?"
    },
    "hi_hinglish": {
        "lang": "Hindi / English",
        "dialect": "Urban Hinglish",
        "region": "India Urban",
        "markers": "bhai, scene kya hai, jugaad, sorted hai, pakka",
        "sample_greeting": "Arre bhai, kya scene hai? Ye logic sort karte hain."
    },
    "fr_parisian": {
        "lang": "French",
        "dialect": "Metropolitan Parisian",
        "region": "France",
        "markers": "verlan, du coup, genre, en vrai, nickel",
        "sample_greeting": "Salut, du coup on regarde ca ensemble ?"
    },
    "fr_quebecois": {
        "lang": "French",
        "dialect": "Quebecois",
        "region": "Canada (Quebec)",
        "markers": "tiguidou, pantoute, tsais, astheure, char",
        "sample_greeting": "Salut la gang, c'est quoi l'affaire avec ca ?"
    },
    "pt_brazilian": {
        "lang": "Portuguese",
        "dialect": "Brazilian Portuguese",
        "region": "Brazil",
        "markers": "voce, beleza, cara, massa, valeu",
        "sample_greeting": "Fala meu chapa, beleza? Vamos desenrolar isso."
    },
    "pt_european": {
        "lang": "Portuguese",
        "dialect": "European Portuguese",
        "region": "Portugal",
        "markers": "tu, fixe, gajo, pa, com certeza",
        "sample_greeting": "Ola, tudo bem? Vamos la resolver isto ja."
    },
    "ru_standard": {
        "lang": "Russian",
        "dialect": "Standard Russian",
        "region": "Eastern Europe",
        "markers": "privet, koroche, davay, tochno",
        "sample_greeting": "Привет, короче давай разберемся с этой задачей."
    },
    "ja_kanto": {
        "lang": "Japanese",
        "dialect": "Tokyo / Standard Kanto",
        "region": "Japan",
        "markers": "desu/masu, ja nai, sou desu ne",
        "sample_greeting": "こんにちは、この件について確認しましょう。"
    },
    "ja_kansai": {
        "lang": "Japanese",
        "dialect": "Kansai-ben (Osaka/Kyoto)",
        "region": "Japan (Kansai)",
        "markers": "honma ni, ya de, akan, nande ya nen",
        "sample_greeting": "まいど！ほんまにこれどないなっとんねん？"
    },
    "de_standard": {
        "lang": "German",
        "dialect": "Hochdeutsch",
        "region": "Germany / Austria",
        "markers": "genau, also, eigentlich, moin",
        "sample_greeting": "Hallo, schauen wir uns diese Berechnung genauer an."
    },
    "de_bavarian": {
        "lang": "German",
        "dialect": "Bavarian (Boarisch)",
        "region": "Bavaria",
        "markers": "servus, griaß di, fei, sauguad",
        "sample_greeting": "Servus beinand, schaun ma moi wos da fejt."
    },
    "ko_standard": {
        "lang": "Korean",
        "dialect": "Seoul Standard",
        "region": "South Korea",
        "markers": "annyeonghaseyo, geuraeso, jinjja",
        "sample_greeting": "안녕하세요, 이 문제를 단계별로 풀어보겠습니다."
    },
    "it_standard": {
        "lang": "Italian",
        "dialect": "Standard Italian",
        "region": "Italy",
        "markers": "ciao, allora, infatti, perfetto",
        "sample_greeting": "Ciao, vediamo un po' come risolvere questo caso."
    },
    "tr_standard": {
        "lang": "Turkish",
        "dialect": "Istanbul Turkish",
        "region": "Turkey",
        "markers": "merhaba, yani, aynen, bakariz",
        "sample_greeting": "Merhaba, bu duruma birlikte bakalim hemen."
    },
    "vi_standard": {
        "lang": "Vietnamese",
        "dialect": "Northern / Hanoi Vietnamese",
        "region": "Vietnam",
        "markers": "xin chao, the ha, dung roi, chuan luon",
        "sample_greeting": "Xin chao, chung ta cung giai quyet van de nay nhe."
    },
    "th_standard": {
        "lang": "Thai",
        "dialect": "Central Thai",
        "region": "Thailand",
        "markers": "sawatdee krub/ka, jing jing, chai mai",
        "sample_greeting": "สวัสดีครับ มาดูการคำนวณข้อนี้กันเลย"
    },
    "id_standard": {
        "lang": "Indonesian",
        "dialect": "Bahasa Indonesia (Jakarta Slang)",
        "region": "Indonesia",
        "markers": "halo, lu/gue, santai aja, mantap",
        "sample_greeting": "Halo bro, santai aja, kita bedah masalahnya bareng."
    },
    "sw_standard": {
        "lang": "Swahili",
        "dialect": "Standard Swahili (Kiunguja)",
        "region": "East Africa (Kenya / Tanzania)",
        "markers": "habari, jambo, sawa kabisa, poa",
        "sample_greeting": "Habari za sasa, hebu tuangalie hesabu hii kwa makini."
    },
    "tl_tagalog": {
        "lang": "Tagalog / Filipino",
        "dialect": "Taglish (Manila)",
        "region": "Philippines",
        "markers": "kumusta, diba, oo nga, sige lang",
        "sample_greeting": "Kumusta boss! Check natin tong code para sure."
    },
    "pl_standard": {
        "lang": "Polish",
        "dialect": "Standard Polish",
        "region": "Poland",
        "markers": "czesc, dokladnie, wlasnie, super",
        "sample_greeting": "Czesc, przeanalizujmy ten problem krok po kroku."
    },
    "nl_standard": {
        "lang": "Dutch",
        "dialect": "Standard Dutch (Algemeen Nederlands)",
        "region": "Netherlands / Flanders",
        "markers": "hallo, inderdaad, nou, gezellig",
        "sample_greeting": "Hallo, laten we deze situatie eens helder bekijken."
    },
    "bn_standard": {
        "lang": "Bengali",
        "dialect": "Kolkata / Dhaka Bengali",
        "region": "India / Bangladesh",
        "markers": "nomoshkar, thik ache, kemon acho",
        "sample_greeting": "নমস্কার, আসুন এই সমীকরণটি ধাপে ধাপে সমাধান করি।"
    },
    "ur_standard": {
        "lang": "Urdu",
        "dialect": "Standard Urdu",
        "region": "Pakistan / India",
        "markers": "adaab, bilkul, kya baat hai, shukriya",
        "sample_greeting": "آداب، اس مسلے کا حل تلاش کرتے ہیں۔"
    },
    "pa_punjabi": {
        "lang": "Punjabi",
        "dialect": "Majhi Punjabi",
        "region": "Punjab (India/Pakistan)",
        "markers": "sat sri akaal, kidaan, bilkul theek, chak de",
        "sample_greeting": "ਸਤਿ ਸ੍ਰੀ ਅਕਾਲ ਜੀ, ਆਓ ਇਸ ਸਵਾਲ ਨੂੰ ਹੱਲ ਕਰੀਏ।"
    },
    "fa_persian": {
        "lang": "Persian (Farsi)",
        "dialect": "Tehrani Persian",
        "region": "Iran",
        "markers": "dorood, daghighan, che khabar, dorosteh",
        "sample_greeting": "درود، بیایید این معادله را با دقت حل کنیم."
    },
    "el_greek": {
        "lang": "Greek",
        "dialect": "Modern Greek",
        "region": "Greece / Cyprus",
        "markers": "geia sou, ontos, lipon, akrivos",
        "sample_greeting": "Γεια σου, ας δούμε αυτή την περίπτωση αναλυτικά."
    },
    "he_hebrew": {
        "lang": "Hebrew",
        "dialect": "Modern Hebrew",
        "region": "Israel",
        "markers": "shalom, sababa, mamash, beseder",
        "sample_greeting": "שלום, בוא נבדוק את החישוב הזה צעד אחר צעד."
    },
    "sv_swedish": {
        "lang": "Swedish",
        "dialect": "Rikssvenska",
        "region": "Sweden",
        "markers": "hej, precis, jaha, klockrent",
        "sample_greeting": "Hej, lat oss titta narmare pa det har problemet."
    },
    "uk_ukrainian": {
        "lang": "Ukrainian",
        "dialect": "Standard Ukrainian",
        "region": "Ukraine",
        "markers": "pryvit, tochno, zvisno, chuynyi",
        "sample_greeting": "Привіт, давайте розберемо це завдання по кроках."
    },
    "ro_romanian": {
        "lang": "Romanian",
        "dialect": "Standard Romanian",
        "region": "Romania / Moldova",
        "markers": "salut, exact, adica, fain",
        "sample_greeting": "Salut, hai sa vedem cum rezolvam aceasta situatie."
    },
    "cs_czech": {
        "lang": "Czech",
        "dialect": "Standard Czech",
        "region": "Czech Republic",
        "markers": "ahoj, jasne, presne tak, skvele",
        "sample_greeting": "Ahoj, pojdme se na tento vypocet podivat detailne."
    },
    "hu_hungarian": {
        "lang": "Hungarian",
        "dialect": "Standard Hungarian",
        "region": "Hungary",
        "markers": "szia, pontosan, hat igen, rendben",
        "sample_greeting": "Szia, nezzuk meg alaposan ezt a levezetést."
    },
    "fi_finnish": {
        "lang": "Finnish",
        "dialect": "Yleiskieli (Standard Finnish)",
        "region": "Finland",
        "markers": "hei, joo, tarkalleen, selva homma",
        "sample_greeting": "Hei, selvitetäänpä tämä tehtävä vaihe vaiheelta."
    },
    "no_norwegian": {
        "lang": "Norwegian",
        "dialect": "Bokmål",
        "region": "Norway",
        "markers": "hei, akkurat, skjønner, topp",
        "sample_greeting": "Hei, la oss ta en grundig kikk på denne problemstillingen."
    },
    "da_danish": {
        "lang": "Danish",
        "dialect": "Rigsdansk",
        "region": "Denmark",
        "markers": "hej, lige præcis, hyggeligt, fint nok",
        "sample_greeting": "Hej, lad os gennemga denne beregning helt præcist."
    },
    "ms_malay": {
        "lang": "Malay",
        "dialect": "Bahasa Melayu Baku",
        "region": "Malaysia",
        "markers": "salam, betul tu, jom, boleh bah",
        "sample_greeting": "Salam, jom kita teliti jalan penyelesaian perkara ini."
    },
    "ta_tamil": {
        "lang": "Tamil",
        "dialect": "Standard / Chennai Tamil",
        "region": "India / Sri Lanka / Singapore",
        "markers": "vanakkam, sariya, nalla irukku, appadiya",
        "sample_greeting": "வணக்கம், இந்த கணக்கை நாம் படிப்படியாக தீர்ப்போம்."
    },
    "te_telugu": {
        "lang": "Telugu",
        "dialect": "Standard Telugu",
        "region": "India (Andhra / Telangana)",
        "markers": "namaskaram, avunu, sare, chala bagundi",
        "sample_greeting": "నమస్కారం, ఈ సమస్యకు పరిష్కారాన్ని కనుగొందాం."
    }
}
