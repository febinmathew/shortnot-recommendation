??
??
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( ?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
?
HashTableV2
table_handle"
	containerstring "
shared_namestring "!
use_node_name_sharingbool( "
	key_dtypetype"
value_dtypetype?
.
Identity

input"T
output"T"	
Ttype
w
LookupTableFindV2
table_handle
keys"Tin
default_value"Tout
values"Tout"
Tintype"
Touttype?
b
LookupTableImportV2
table_handle
keys"Tin
values"Tout"
Tintype"
Touttype?
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
?
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( ?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
E
Relu
features"T
activations"T"
Ttype:
2	
?
ResourceGather
resource
indices"Tindices
output"dtype"

batch_dimsint "
validate_indicesbool("
dtypetype"
Tindicestype:
2	?
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ??
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.10.02v2.10.0-rc3-6-g359c3cdfc5f8ϖ
?	
ConstConst*
_output_shapes	
:?*
dtype0	*?	
value?	B?		?"?       	       
                                                                                                                        !       "       #       $       %       &       '       (       )       *       +       ,       -       .       /       0       2       5       6       7       8       9       :       ;       <       =       >       ?       @       A       B       C       D       E       F       G       H       I       J       K       L       M       N       O       P       Q       R       S       T       U       V       W       X       Y       Z       [       \       ]       ^       _       `       a       b       c       d       e       f       g       h       i       j       k       l       m       n       o       p       q       r       s       t       u       v       w       x       y       z       {       |       }       ~              ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       
?	
Const_1Const*
_output_shapes	
:?*
dtype0	*?	
value?	B?		?"?                                                        	       
                                                                                                                                                                  !       "       #       $       %       &       '       (       )       *       +       ,       -       .       /       0       1       2       3       4       5       6       7       8       9       :       ;       <       =       >       ?       @       A       B       C       D       E       F       G       H       I       J       K       L       M       N       O       P       Q       R       S       T       U       V       W       X       Y       Z       [       \       ]       ^       _       `       a       b       c       d       e       f       g       h       i       j       k       l       m       n       o       p       q       r       s       t       u       v       w       x       y       z       {       |       }       ~              ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       
?
Const_2Const*
_output_shapes
:~*
dtype0	*?
value?B?	~"?                                                        	       
                                                                                                                                                                  !       "       #       $       %       &       '       (       )       *       +       ,       -       .       /       0       1       2       3       4       5       6       7       8       9       :       ;       <       =       >       ?       @       A       B       C       D       E       F       G       H       I       J       K       L       M       N       O       P       Q       R       S       T       U       V       W       X       Y       Z       [       \       ]       ^       _       `       a       b       c       d       e       f       g       h       i       j       k       l       m       n       o       p       q       r       s       t       u       v       w       x       y       z       {       |       }       ~       
??
Const_3Const*
_output_shapes
:~*
dtype0*??
value??B??~B B
#developer #comedy #codinglifeB( #Asgar  #enjoy  #funny  #video  #swag
 B #ForYou  #Bangladesh B> #enjoy  #attitude  #video  #funny  #💘...outfitinspiration B9 #enjoy  #funny  #photography  #swag
  #video  #attitude B #funny  #Asgar  #reels B- #funny  #enjoy  #Asgar  #Hastags  #dancing
 B# #op #famous #creator op bhabhi ji BD #मेहनत का फल देगा ऊपर वाला B"I see nothing"B#Asgar B#ReelKaroFeelKaroB?#instamodel #saree #Soniyaa3366 #Sareegirl #Blacksaree #pinklehenga #tgirls #transgirls #SouthIndianActress #BridalSaree #Bridalshoot #silksaree #sareeblouse #blouse #boytogirl #maletofemale #transwoman #SouthIndianBride #sareeindia #sareeswag #Bluesaree #sareelove #lehengacholi #lehenga #salwarsuits #boytogirltransformationB#op #bhabhi ji B#seavibes #funtime #sunsetB#travel #india BJ*Books flight to Santorini* ✈️🇬🇷

✍️ Twitter/mavericksmoviesB?@daviddobrik Crashes Bugatti😬 

• Bugatti Chiron 

🚀 Follow -》 @automotive_escape for daily car content! 

🎥 -》 Dm or Tag Below 
@the.hamilton.collection @bugatti

-------------------------------------

#supercar #bugatti #supercars #supercarsoflondon #supercarspotting #supercarsdaily #bugattichiron #supercarsofinstagram #supercarspotting #supeedaily #supercars247 #bugattidivo #supercarblondie #bugattilavoiturnoire #bugattichirons #bugattis #instasupercar #igsupercars #londonsupercars #supercarslondon #supercarsunday #bugattiveyron #supercarclub #supercarlife #supercarlovers #luxuryB[@karmathelekhak @kalamkaarmusic @deepkalsimusic 

#lenameranaam #dehradunkamerakhoon #reelsBs@shemaroobhakti @shankar.mahadevan #devahodeva #shemaroobhakti
#shankarmahadevan #ganpatibappamorya #ganpati #reelsBg@timesmusichub @jungleeemusic
#followfollow #nannakuprematho #jrntr #rakulpreet #rakulpreetsingh #reelsB?A true inspiration to so many! 👸 From changing the law to make the monarchy more equal for women to being the only female royal in the Armed forces - her Majesty Queen Elizabeth II made history as a female icon on numerous occasions 💪

She will be so missed - RIP Queen Elizabeth II 🕊B?After the first episode of MAFS 2022 aired last night, we couldn’t help but take a look back on who’s still together from last year’s series… 👀BAsar yeh naya hai💚😙

#reel #reels #reelitfeelit #reelkarofeelkaro #ketakikulkarni #ketakians #explore #explorepage #viralB?Audi R8 V10 Quad Bike 😱 

• Engler Audi R8 V10 Quad Bike

🚀 Follow -》 @automotive_escape for daily car content!

📷 -》 @tfjj
@englerautomotive

-----------------------------------------

#carswithoutlimit #audir8 #audir8v10 #amazingcars247 #carsfascinateB?Audi RS6 Mansory 🔥

🚀 Follow -》 @automotive_escape for daily car content!

🎥 -》@cars_madrid_
@audiofficial

-----------------‐-------------------

#supercarsunday #audi #rs6 #audirs6 #mansoryB&Bangla rap song  #Bangladesh  #short  B?Banja tu mera humsafar❤️

#reel #reels #reelitfeelit #reelkarofeelkaro #ketakikulkarni #ketsfam #ketakians #viral #explore #explorepageB?Basic things 💫

Et tombée par hasard sur @wawapod

.

#outfitinspiration #outfitinspo 
#outfitoftheday #discoverunder20k #discover100k 
#classyoutfit #ａｅｓｔｈｅｔｉｃ 
#bordeaux #parisianvibe #everydayoutfit
#bloggerstyle 
#styleinspiration #look
#contentcreator #frenchgirlstyle #dunklow #nikedunklow 
#frenchmood #frenchgirlstyle #ootd #parisianstyle #reeloutfitB?Brabus Porsche 😎

🚀 Follow -》 @automotive_escape for daily car content!

🎥 -》 @mr_automotive
@porsche @theofficialbrabus 

-------------------------------------

#porsche #porsche911 #brabus #brabusporsche #amazingcars247B?Britney’s vibes 🧸

#outfitinspiration #outfitinspo 
#outfitoftheday #discoverunder20k #discover100k 
#classyoutfit #ａｅｓｔｈｅｔｉｃ 
#bordeaux #parisianvibe #everydayoutfit
#bloggerstyle #ootd #nike #dunklowpanda 
#styleinspiration #styleinfluencer
#contentcreator #frenchgirlstyle #zaraoutfit 
#frenchmood #frenchgirlstyle #photoeverydayB?Candy Apple Red Koenigsegg Regera 🥵

• Koenigsegg Regera 

🚀 Follow -》 @automotive_escape for daily car content!

🎥 -》 @tfjj
jimmy.fulltime @horsepowerracinguk 
@koenigseggB?Caption this! 😎

• Ferrari LaFerrari
• Mclaren P1
• Porsche 918 Spyder

🚀 Follow -》 @automotive_escape for daily car content!

🎥 - @supercardriver @blue_chip_car_collector
@ferrari @mclaren @porsche

-------------------------------------

 #carspotted #p1 #918 #laferrari #HypercarB?Cheapest cars for teens🙃🚗💨

🚀 Follow -》 @automotive_escape for daily car content!

🎥 - @flaexus

-------------------------------------

#hypercarsdaily #mercedes #amg #amgperformance #paganihuayra #bugatti #bugattichiron #ferrari #ferrarilaferrariB?Dancing on beats 🎶

#instamodel #saree #Soniyaa3366 #Sareegirl #Blacksaree #pinklehenga  #SouthIndianActress #BridalSaree #Bridalshoot #silksaree #sareeblouse #blouse #boytogirl  #SouthIndianBride #sareeindia #sareeswag #Bluesaree #sareelove #lehengacholi #lehenga #salwarsuits #croptop #Tomboy #instareels #reelkarofeelkaroBnDoggo's really saying: "Keep Calm! How many tails wagging can you see?!" 🤣🐾⁠
⁠
🎥 @jaleise_⁠
⁠BDrama queen 🤣B?Escapade ensoleillée et pleine de love 💕
.
.
#couplegoals #couple #couplegoal #areformidable #youarethebest #fyp #viral #outfitinspiration #lifestylemodel #thatgirl #thatgirlaesthetic #onveutduvrai #womanrights #lestyleàlafrançaise #mode #tendance #frenchgirlstyle #couplegoals #photography #brownhair #southoffrance #styleminimalist #collioure #perpignanB?Felt cute about this reel idk why🤭❤️

Top @igclothing.co 
Location @naturesdreamland_holidayhomes 

#reels #reel #reelkarofeelkaro #reelitfeelit #ketakikulkarni #ketsfam #ketakians #exploreme #explorepage #exploreB?Fireworks 🔥🔥

• Lamborghini Aventador SVJ

🚀 Follow -》 @automotive_escape for daily car content!

🎥 -》 @mikeynoga 
@grassgassass @lamborghini

-------------------------------------

#lamborghini #lambocardaily #superexoticscars
#lamborghiniaventadorsvj #luxurycarlife
#cars4life #carslife #aventador #aventadorsvj 
#lamborghinilover #svj
#lamborghiniaventadorBFunny VideosB?Gonna get dressed for success

#instamodel #saree #Soniyaa3366 #Sareegirl #hollywood #pinklehenga #transgirls #SouthIndianActress #BridalSaree #Bridalshoot #silksaree #sareeblouse #blouse #boytogirltransformation  #SouthIndianBride #sareeindia #sareeswag #Bluesaree #sareelove #lehengacholi #lehenga #salwarsuits #sareewoman #sareegirl #instasaree  #dressesonline #dressesB?Hangover 💚

Location @naturesdreamland_holidayhomes 
Wearing @igclothing.co 

#reels #reel #reelitfeelit #reelkarofeelkaro #ketakikulkarni #ketakians #foryoupage #explorepage #exploreB?Happy Independence Day everyone 🇮🇳🧡

#75thindependenceday #Independenceday #Indepenceday2022 #India #Meradeshmahan #harghartiranga #reels #ketakikulkarni #ketakiansB?Happy Janmashtami everyone 🦚💗🤗
Tried something new for Janmashtami, hope you all love it 🫶🏻

Thankyou @salomi491 for this look😍

#reels #reel #reelitfeelit #reelkarofeelkaro #janmashtami #krishnajanmashtami #krishna #radhakrishna #radha #explorepage #ketakikulkarni #ketakiansB?Haye jiska dil hua💛🫶🏻

#reels #reel #reelitfeelit #reelkarofeelkaro #ketakikulkarni #ketakians #ketsfam #explorepage #exploreB?Her voicee🥹🤌🏻

#reel #reels #reelitfeelit #reelkarofeelkaro #instagram #instagood #ketakikulkarni #ketakians #viral #explore #explorepageB[Howling 🤣 It was the “DiCaprioOoOoOo” that got me 😭

🎥 @NBC ✍️ @phil.lewisB?I am all set for Ganesh Chaturthi with Saregama's #CarvaanMini Ganesh. Carvaan Mini Ganesh comes pre-loaded with 300 Ganesh Mantras, Poojas, Marathi and Hindi Bhajans, Katha and Aartis. The pre-loaded devotional songs are dedicated to #LordGanesha.

Buy yours at saregama.com

#ganeshchaturthi
#ganeshutsav #ganeshotsav #ganeshfestival #happyganeshchaturthi #ganeshji #CarvaanMiniGanesh #Saregama #promotion #ketakikulkarni #reelsB?Iced latte please ☕️
.
.
.
#ootd #grwm #outfitlove #frenchgirl #frenchvibes #parisianstyle #jean #mocassins #starbucks #icedlatte #discoverunder100k #frenchstyle #lookofday #parisianlifestyle #outfitinspiration #outfitofthedayB?If my husband does not whip out our secret handshake at our wedding I don't want him 🤝⁠
⁠
🎥 @aaronxaquino x @nicolerosoveB?Inki baatein kabhi puri nhi hogi @ritumaan01 @hairbysuvarna 😵‍💫🤦🏻‍♀️

#ketakikulkarni #ketakians #ketsfam #reels #reel #reelitfeelit #explore #explorepage #vanitydiariesB?Je me sentais très mariée ou petit ange dans cette robe, mais je me sentais bien 🕊

.

#outfitinspiration #outfitinspo 
#outfitoftheday #discoverunder20k #discover100k 
#classyoutfit #ａｅｓｔｈｅｔｉｃ 
#bordeaux #parisianvibe #everydayoutfit
#bloggerstyle 
#styleinspiration #styleinfluencer
#contentcreator #frenchgirlstyle #zaraoutfit 
#frenchmood #frenchgirlstyle #pinterestaesthetic #photoeveryday #reeloutfitB?Kabhi nahi jo kaha hai❤️🤭

Ps: idk what’s going on the background 🥲

#reels #reel #reelitfeelit #reelviral #viral #ketakikulkarni #ketsfam #ketakians #explorepage #exploreB?Kanha 🥹🦚🙏🏻

#kanha #krishna #krishnajanmashtami #radha #radhakrishna #ketakikulkarni #ketsfam #ketakians #reels #reelsviral #explorepageB?Kese kab hogaya🤭💚

#reels #reel #reelkarofeelkaro #reelit #ketakikulkarni #ketsfam #ketakians #exploreme #explorepage #exploreB?Kisi ka toh hoga hi tu❤️

📍 @naturesdreamland_holidayhomes 

#reels #reel #reelitfeelit #reelkarofeelkaro #ketakikulkarni #ketsfam #ketakians #foryou #explore #explorepageBXLaung laachi 2.0💕
@ammyvirk
@amberdeepsingh
@neerubajwa
@burfimusic
#promotion #reelsB?Le sud ✨
C’est vraiment un bonheur d’entendre chanter les
Cigales ✨🫶🏼

#provence #cigales #sud #suddelafrance #marseille #provencealpescotedazur #aesthetic #dreamhouse #aestheticvideos #aestheticview #windowlight #windowB?Le sud 🦋

.

#outfitinspiration #outfitinspo 
#outfitoftheday #discoverunder20k #discover100k 
#classyoutfit #ａｅｓｔｈｅｔｉｃ 
#bordeaux #parisianvibe #everydayoutfit
#bloggerstyle 
#styleinspiration #styleinfluencer
#contentcreator #frenchgirlstyle #zaraoutfit 
#frenchmood #frenchgirlstyle #pinterestaesthetic #photoeveryday #reeloutfitBoLet’s all remember how amusing the Queen was ❤️

RIP Queen Elizabeth II 👑

🎥 TikTok/british.royaltyB?L’effet Barbie shoes 💘

.
.
.

#outfitinspiration #outfitinspo 
#outfitoftheday #discoverunder20k #discover100k 
#classyoutfit #ａｅｓｔｈｅｔｉｃ 
#bordeaux #parisianvibe #everydayoutfit
#bloggerstyle 
#styleinspiration #styleinfluencer
#contentcreator #frenchgirlstyle #bershka 
#frenchmood #frenchgirlstyle #pinterestaesthetic #photoeveryday #reeloutfit #barbiestyle #barbiegirlBMe during any social gatheringB?Meghan Markle has been praised for making a sweet comment to a royal aide after greeting some of Queen Elizabeth’s mourners 💐

Prince Harry’s wife made the comment as she completed a walkabout outside Windsor Castle with her husband where they greeted royal fans.

Many of the fans had brought flowers to pay tribute to the late Queen, who passed away on Thursday (September 8) at the age of 96.

One of the well wishers gave their flowers to Meghan, who promised that she would lay them at a gate on their behalf.

When a Royal aide offered to do it for Meghan, she politely declined and said that she wanted to keep her word to the mourners ❤️B?Mercedes AMG GT Black Series 🚀

🚀 Follow -》 @automotive_escape for daily car content!

🎥 - @arndt_autovermietung
@g.wizzle_  @mercedesamg

-------------------------------------

#supercar #cars #amg #mercedes #mercedesamg #amgperformanceB?Milkshake rafraîchissant 🍌💕

Avec ce temps incroyable j’adore faire des smoothie et milkshake 🤍
Rien de plus simple, gourmand et rafraîchissant 

Ingrédients pour 2 : 
1 banane
1 grosse poignée de fruits rouges @picardsurgeles 
1 boule de glace au yaourt ( ou vanille ) @picardsurgeles 
1 grand verre de lait 

#milk #smoothie #iced #milkshake #aesthetic #icecream #photography #icecreamlover #icedfruit #banana #bananalovers #fruitsrouges #fit #healthyfood #healthyrecipes #reelsrecipe #fypB?Même si c’était pas ma pizza : vous savez que c’est une victoire pour moi 🫶🏼🥹

.
.
#pizzanapoletana #pizzalovers #pizzalover 
#ａｅｓｔｈｅｔｉｃ 
#parisianvibe 
#bloggerstyle #ootd #contentcreator 
#styleinspiration #styleinfluencer 
#discoverunder30k #discover100k 
#contentcreator #frenchgirlstyle 
#frenchmoodB%Nature in its beauty #nature #beauty B?Night out avec une chemise d’homme et des talons très hauts 💫

.

#outfitinspiration #outfitinspo 
#outfitoftheday #discoverunder20k #discover100k #ootd 
#classyoutfit #ａｅｓｔｈｅｔｉｃ 
#bordeaux #parisianvibe #everydayoutfit
#bloggerstyle #ootd 
#styleinspiration #styleinfluencer
#contentcreator #frenchgirlstyle #talonshauts #talonsaiguilles 
#frenchmood #frenchgirlstyle #photoeveryday #highheelsBBNow THAT is my kinda wedding 🥔🍸⁠
⁠
🎥 @cookinwithboozeBzOMG Halle Bailey as Ariel 😍 Going to have “Part of Your World” running around my head all day 🤣👏

🎥 DisneyB?Okay ik I’m late but how can I not post a transition video😭💖 

#transition #transitionreels #transitionvideo #reels #reel #reelitfeelit #ketakikulkarni #ketsfam #ketakians #radhakrishna #krishna #radhaB?Old Hindi songs>>>> 

Outfit by @dressbollywood27 🤍

#reels #reel #reelkarofeelkaro #reelitfeelit #ketakikulkarni #ketakians #explore #explorepage #foryouB?On my way 🧸💼

#outfitinspiration #outfitinspo 
#outfitoftheday #ootd #discover100k 
#classyoutfit #ａｅｓｔｈｅｔｉｃ 
#bordeaux #parisianvibe #everydayoutfit
#bloggerstyle #ootd #mannequin 
#styleinspiration #styleinfluencer
#contentcreator #frenchgirlstyle 
#frenchmood #frenchgirlstyleB?One of my favorites

#instamodel #saree #Soniyaa3366 #Sareegirl #Blacksaree #pinklehenga #tgirls #transgirls #SouthIndianActress #BridalSaree #Bridalshoot #silksaree #sareeblouse #blouse #boytogirl #maletofemale #transwoman #SouthIndianBride #sareeindia #sareeswag #Bluesaree #sareelove #lehengacholi #lehenga #salwarsuits #boytogirltransformation #bollywoodhot #bollywoodstyle #IndianFashion #actressmemesB?Outfit Check 
.
.
.
.
.
.

#outfitinspos #90s #ootd #vintagelook
#ootd #outfitoftheday #discoverunder100k #wiwt 
#70slook #outfitcheck #classyoutfit #jonakgirls 
#parisianstyle #parisianvibe #everydayoutfit #mangogirls 
#contentcreator #ootdshare #luxurylifestyle
#styleinspiration #styleinfluencer
#howtobeparisian #frenchgirlstyle #parisienne
#whowhatwear #frenchmoodBJPeaceful places😌🫶🏻✨

#reels #reel #reelitfeelit #ketakikulkarniB?Popcorn machine 🍿

• Ferrari SF90

🚀 Follow -》 @automotive_escape for daily car content!

🎥 -》 @hdtuningaz
@ferrari

-------------------------------------

#ferrari #enzoferrari #ferraritestarossa #supercar #carswithoutlimits #ferrari812superfast #ferrari458  #ferrari458spider #ferrariroma #ferrarisf90stradale #ferrari488pistaspider #488pistaspider #488pistaB?Pyaar hojayegaaa🤍🫣

#reels #reel #reelkarofeelkaro #reelitfeelit #trending #ketakikulkarni #ketsfam #ketakians #viral #viralreel #explore #explorepageB?Recap summer 🥺💫

.
.

#outfitinspiration #outfitinspo 
#outfitoftheday #discoverunder20k #discover100k 
#classyoutfit #ａｅｓｔｈｅｔｉｃ 
#marseille #parisianvibe #everydayoutfit
#bloggerstyle #southoffrance 
#styleinspiration #styleinfluencer
#contentcreator #frenchgirlstyle #dosnu #summer2022 #memorie 
#frenchmood #frenchgirlstyle #photoeverydayB?Red Beast waiting for its owner in Monaco 😎

• Ferrari SF90 Stradale

🚀 Follow -》 @automotive_escape for daily car content!

🎥 -》 @aa.carsandexotics
@ferrari

-------------------------------------

#ferrari #sf90 #stradale #ferrarisf90stradale #ferrarisf90 #automotive #supercar #hypercar  #carlifestyle #luxurycars #racing #racecar  #carspotting #carB?Reel Karo Feel Karo ✌️

#instamodel #saree #Soniyaa3366 #Sareegirl #Blacksaree #pinklehenga #tgirls #transgirls #SouthIndianActress #BridalSaree #Bridalshoot #silksaree #sareeblouse #blouse #boytogirl #maletofemale #transwoman #SouthIndianBride #sareeindia #sareeswag #Bluesaree #sareelove #lehengacholi #lehenga #salwarsuits #boytogirltransformationB?Regera Transformer in Dubai ⚡️ 

• Koenigsegg Regera

🚀 Follow -》 @automotive_escape for daily car content!

🎥 -》 @karim.luxury 
@alqalammotorz @koenigsegg 

-------------------------------------

#carswithoutlimit #koenigseggregistry #regera #koenigseggregera #hypercar #hypercarsdailyB?Remembering #PrincessDiana today 🕊 Marking 25 years since she tragically passed away, here are some facts you may not have known about the late #PrincessofWales ❤️B?SPOTTED👀: Flame Spitting Centenario 

• Lamborghini Centenario

🚀 Follow -》 @automotive_escape for daily car content!

🎥 -》 @lacarspotter_
@aka_boodee @dave_dde @dailydrivenexotics @lamborghini 

-------------------------------------

#Hypercar #lamborghini #lamborghini_daily #centenario #lamborghinicentenarioB?Sachez profiter de chaque instant , libérez-vous de vos contraintes, osez voler sans filet : le bonheur est à ce prix.✨

#summertrends #ootd #aesthetic #dailyoutfit
#ootdshare #ootdmagazine #americanstyle
#frenchgirl #discoverunder50k #outfitinspo
#badestoutfits #clothinspos #summeroutfit
#slyleinfluencer #contentcreator#vintagevibes
#90sstyle #parisfashion #onparledemode #marrakech #marrakesh #marrakeshvibes #bikinidayB*Sandra slayed 🙌💜

✍️ @devrobertsBzSo precious 🥹 The Queen really did love animals 🐮❤️

May she rest in peace with Prince Philip now 🕊

🎥 BBCB?Spaghettis au citron 🍋 

Cette recette ensoleillée et parfumée ravira vos papilles !
N’oubliez pas qu’il faut continuer à manger des féculents même l’été 🫶🏼 c’est important !

Ingrédients pour 1 : 
100 g  de spaghetti
1/2 citron
1/2 oignon rouge
parmesan râpé
1 gousse d’ail
15cl de crème fraiche

Préparation :
1. Faire revenir l’oignon rouge émincé dans une poêle. Mettre ensuite la crème. Hacher l’ail et l’ajouter. 
2. Cuire les pâtes dans une casserole d’eau à ébullition.
3.Sortir les pâtes du feu. Égoutter. Verser la crème. Mélanger.
4.Parsemez de parmesan râpé.Presser le citron. Saler, poivrer .
 
Produit @flinkfrance 

#easyrecipes #pasta #pastalover #crevette #pates #pastarecipe #cheese #healthyrecipes #recette #recettehealthy #recettefacile #recetterapide #lemonpasta #pastalemon #lemonB?Sunday's First Post

#instamodel #saree #Soniyaa3366 #Sareegirl #Blacksaree #pinklehenga #tgirls #transgirls #SouthIndianActress #BridalSaree #Bridalshoot #silksaree #sareeblouse #blouse #boytogirl #maletofemale #transwoman #SouthIndianBride #sareeindia #sareeswag #Bluesaree #sareelove #lehengacholi #lehenga #salwarsuits #boytogirltransformationB?TAKE or PASS!

• Mercedes AMG SL63 

🚀 Follow -》 @automotive_escape for daily car content!

🎥 -》 @mr.benz63
@mercedesbenz 

-------------------------------------

#supercarsforlife #mercedesamg #amgperformance #sl63 #sl63amg #mercedessl63amgB{Tere pyaar mein hojau fanaaaa🤍

#reels #reel #reelitfeelit #ketakikulkarni #ketakians #explorepage #explore #exploremoreB?Tere te hi mardaaa🫶🏻🤭

Wearing @trendy__stuffs18 

#reels #reel #reelitfeelit #reelkarofeelkaro #expressions #lipsync #ketakikulkarni #ketsfam #ketakians #explorepage #exploreB?That's one way to do it 🙌⚡️⁠
⁠
🎥 @michaelandrewrgB?The 1 of 60 Bugatti Chiron Pur Sport, the ultimate nimble track-focused Bugatti!🔥 What color would you get on it?

• Bugatti Chiron Pur Sport

🚀 Follow -》 @automotive_escape for daily car content!

🎥 -》 @supercarheather
@bugatti

-------------------------------------

#Bugatti #Chiron #bugattichirons #bugattichironpursport #chironpursport #HypercarsBHThe Queen truly loved her horses 🥲⁠
⁠
RIP Queen Elizabeth II 👑B?The Queen was adored by so many 🥲💐⁠
⁠
🎥 @_sophie90B?The awkward 2-second wait before grabbing a juice to claim the table while you both go up for food 🤣⁠
⁠
🎥 @letterboxloveofficialB?The brand new Bugatti W16 Mistrale 🔥 

🚀 Follow -》 @automotive_escape for daily car content!

🎥 - @tfjj
@bugatti

-------------------------------------

 #Bugatti #Mistrale #bugattimistral #Hypercar #SupercarsB?The prince of Qatar visiting Monaco in his Bugatti Divo😍😍

• Bugatti Divo

Follow 🚀-》 @automotive_escape for daily car content!

🎥 -》 @niels_carspotting

-------------------------------------

#bugatti #divo #bugattidivoB?This dresssssss 🖤

L’effet CapCut me donne un air vachement plus confiante que dans la vraie vie 🫢

.
.

.

#outfitinspiration #outfitinspo 
#outfitoftheday #discoverunder20k #discover100k 
#classyoutfit #ａｅｓｔｈｅｔｉｃ 
#bordeaux #parisianvibe #everydayoutfit
#bloggerstyle #ootd #zaraoutfits 
#styleinspiration #styleinfluencer
#contentcreator #frenchgirlstyle #zaraoutfit 
#frenchmood #frenchgirlstyle #photoeverydayB?This is why they won Love Island 2022 🙌 @ekinsuofficial walked the Oh Polly catwalk last night showing off her new line with the popular clothing brand and @davidesancli's reaction is everything 🥲❤️

🎥. Twitter/jordanparkererb ✍️ @jayandreasB?This place is everything 💫🤍

Total look Mango 🖤

#outfitinspiration #outfitinspo 
#outfitoftheday #discoverunder30k #discover100k 
#classyoutfit #ａｅｓｔｈｅｔｉｃ #frenchvibes 
#bordeaux #parisianvibe #everydayoutfit
#bloggerstyle #ootd #mangooutfit 
#styleinspiration #styleinfluencer
#contentcreator #frenchgirlstyle #chateau #margaux 
#frenchmood #frenchgirlstyle #photoeverydayBnThis slow mo!🤍

@dhruvmalik_official
@alygoni
@surbhijyoti
@timesmusichub
#promotion #ketakikulkarni #reelsB?This song lives rentless on my mind🥹🤍

#yehishqhai #yehishqhaaye #ketakikulkarni #ketakians #ketsfam #explorepage #explore #reel #reelsB?Thousands of people paid their respects by taking flowers to Buckingham Palace on the evening of the Queen's death 💐 She will be so missed 🕊⁠
⁠
🎥 TikTok/tomiconicB?Tum hi se baat woh keh du💛

#reels #reel #réel #reelsindia #reelsvideo #ketakikulkarni #ketsfam #ketakians #trending #trendingreels #explorepageB?Un matin en France 🫶🏼💫

.
.
.
.

.

#outfitinspiration #outfitinspo 
#outfitoftheday #discoverunder20k #discover100k 
#classyoutfit #ａｅｓｔｈｅｔｉｃ 
#bordeaux #parisianvibe #everydayoutfit
#bloggerstyle 
#styleinspiration #styleinfluencer
#contentcreator #frenchgirlstyle #zaraoutfit #550 #newbalance550 #newbalance #allblackoutfit 
#frenchmood #frenchgirlstyle #pinterestaesthetic #photoeveryday #reeloutfitBViralB?Washing the Beast 🥶

 • Lamborghini Huracan STO

🚀 Follow -》 @automotive_escape for daily car content!

🎥 -》 @incuteag
@lamborghini

-------------------------------------

#lamborghini #huracansto #supertrofeoomologato #supercarsforlife #supercardailyB?We got something 😎

#instamodel #saree #Soniyaa3366 #Sareegirl #Blacksaree #pinklehenga  #transgirls #SouthIndianActress #BridalSaree #Bridalshoot #silksaree #sareeblouse #blouse  #SouthIndianBride #sareeindia #sareeswag #Bluesaree #sareelove #lehengacholi #lehenga #salwarsuits  #bollywoodhot #bollywoodstyle #IndianFashion #tamilponnu #SouthIndianActressB?We’ve all heard the rumours about Harry “spitting” at Chris Pine at the premiere of their new film Don’t Worry Darling in Venice... 🤨

Well here the singer took to the stage on his US tour at New York’s Madison Square Garden and had a little joke about the accusations. 

Chris’ representative said it was “a ridiculous story” and a source close to Harry denied it. 

It comes as people were talking about an alleged feud between the film’s director and Harry’s real girlfriend Olivia Wilde, and his on-screen partner Florence Pugh. 

THE dramaaaa 👀

🎥 StoryFulB?Whistling Turbo by @r1motorsport 😳 

• Porsche 911 992

🚀 Follow -》 @automotive_escape for daily car content!

🎥 -》 @hovpro

------------------------------

#supercar #petrolheadworld #pagani #bugatti #lamborghini #hsdrive  #hypercar #itswhitenoise #car #exoticcar #carenthusiast #drifting #carfanatics #carporn #carswithoutlimits #shift #carinstagram #cargram #superexoticscars #motorsportB<Who said romance was dead?! 🦷🪥

🎥 TikTok/jewels__10Bishq ka gurur #trendingreels B-ma bap ki sewa 🙏
#trending #trendingreels Bmehnat ka phal #trendingreels B]“It’s good - cuz they’ll be back together” 😭 Melted my heart 🥹

🎥 @cewilsonxB❤️B👸B?📲 Aaj Ka Status

Topic : *Halal O Haram Ki Tameez*

Click to watch on YouTube
https://youtube.com/shorts/XV-PGMTXNZk?feature=shareB📽️ Credit: @kingtazdmeB😉B🤣🤣🤣🤣🤣🤣B9🧡✨

#reels #reel #ketakikulkarni #ketakians #ketsfam
?
Const_4Const*
_output_shapes
:#*
dtype0	*?
value?B?	#"?                                                        	       
                                                                                                                                                                  !       "       #       
?
Const_5Const*
_output_shapes
:#*
dtype0	*?
value?B?	#"?                     #       %       &       '       (       *       +       -       .       /       0       1       2       3       4       5       6       7       8       9       :       ;       <       >       ?       @       A       B       C       D       E       F       
I
Const_6Const*
_output_shapes
: *
dtype0	*
value	B	 R 
I
Const_7Const*
_output_shapes
: *
dtype0	*
value	B	 R 
I
Const_8Const*
_output_shapes
: *
dtype0	*
value	B	 R 
?
 Adagrad/dense_2/bias/accumulatorVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" Adagrad/dense_2/bias/accumulator
?
4Adagrad/dense_2/bias/accumulator/Read/ReadVariableOpReadVariableOp Adagrad/dense_2/bias/accumulator*
_output_shapes
:*
dtype0
?
"Adagrad/dense_2/kernel/accumulatorVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*3
shared_name$"Adagrad/dense_2/kernel/accumulator
?
6Adagrad/dense_2/kernel/accumulator/Read/ReadVariableOpReadVariableOp"Adagrad/dense_2/kernel/accumulator*
_output_shapes

:@*
dtype0
?
 Adagrad/dense_1/bias/accumulatorVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*1
shared_name" Adagrad/dense_1/bias/accumulator
?
4Adagrad/dense_1/bias/accumulator/Read/ReadVariableOpReadVariableOp Adagrad/dense_1/bias/accumulator*
_output_shapes
:@*
dtype0
?
"Adagrad/dense_1/kernel/accumulatorVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?@*3
shared_name$"Adagrad/dense_1/kernel/accumulator
?
6Adagrad/dense_1/kernel/accumulator/Read/ReadVariableOpReadVariableOp"Adagrad/dense_1/kernel/accumulator*
_output_shapes
:	?@*
dtype0
?
Adagrad/dense/bias/accumulatorVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*/
shared_name Adagrad/dense/bias/accumulator
?
2Adagrad/dense/bias/accumulator/Read/ReadVariableOpReadVariableOpAdagrad/dense/bias/accumulator*
_output_shapes	
:?*
dtype0
?
 Adagrad/dense/kernel/accumulatorVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@?*1
shared_name" Adagrad/dense/kernel/accumulator
?
4Adagrad/dense/kernel/accumulator/Read/ReadVariableOpReadVariableOp Adagrad/dense/kernel/accumulator*
_output_shapes
:	@?*
dtype0
?
*Adagrad/embedding_1/embeddings/accumulatorVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *;
shared_name,*Adagrad/embedding_1/embeddings/accumulator
?
>Adagrad/embedding_1/embeddings/accumulator/Read/ReadVariableOpReadVariableOp*Adagrad/embedding_1/embeddings/accumulator*
_output_shapes

: *
dtype0
?
(Adagrad/embedding/embeddings/accumulatorVarHandleOp*
_output_shapes
: *
dtype0*
shape
:$ *9
shared_name*(Adagrad/embedding/embeddings/accumulator
?
<Adagrad/embedding/embeddings/accumulator/Read/ReadVariableOpReadVariableOp(Adagrad/embedding/embeddings/accumulator*
_output_shapes

:$ *
dtype0
k

hash_tableHashTableV2*
_output_shapes
: *
	key_dtype0	*
shared_name104*
value_dtype0	
l
hash_table_1HashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name82*
value_dtype0	
l
hash_table_2HashTableV2*
_output_shapes
: *
	key_dtype0	*
shared_name60*
value_dtype0	
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
~
Adagrad/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdagrad/learning_rate
w
)Adagrad/learning_rate/Read/ReadVariableOpReadVariableOpAdagrad/learning_rate*
_output_shapes
: *
dtype0
n
Adagrad/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdagrad/decay
g
!Adagrad/decay/Read/ReadVariableOpReadVariableOpAdagrad/decay*
_output_shapes
: *
dtype0
l
Adagrad/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_nameAdagrad/iter
e
 Adagrad/iter/Read/ReadVariableOpReadVariableOpAdagrad/iter*
_output_shapes
: *
dtype0	
p
dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_2/bias
i
 dense_2/bias/Read/ReadVariableOpReadVariableOpdense_2/bias*
_output_shapes
:*
dtype0
x
dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*
shared_namedense_2/kernel
q
"dense_2/kernel/Read/ReadVariableOpReadVariableOpdense_2/kernel*
_output_shapes

:@*
dtype0
p
dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedense_1/bias
i
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes
:@*
dtype0
y
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?@*
shared_namedense_1/kernel
r
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel*
_output_shapes
:	?@*
dtype0
m

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_name
dense/bias
f
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes	
:?*
dtype0
u
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@?*
shared_namedense/kernel
n
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_output_shapes
:	@?*
dtype0
?
embedding_2/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	? *'
shared_nameembedding_2/embeddings
?
*embedding_2/embeddings/Read/ReadVariableOpReadVariableOpembedding_2/embeddings*
_output_shapes
:	? *
dtype0
?
embedding_1/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *'
shared_nameembedding_1/embeddings
?
*embedding_1/embeddings/Read/ReadVariableOpReadVariableOpembedding_1/embeddings*
_output_shapes

: *
dtype0
?
embedding/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape
:$ *%
shared_nameembedding/embeddings
}
(embedding/embeddings/Read/ReadVariableOpReadVariableOpembedding/embeddings*
_output_shapes

:$ *
dtype0
r
serving_default_user_idPlaceholder*#
_output_shapes
:?????????*
dtype0	*
shape:?????????
|
!serving_default_video_descriptionPlaceholder*#
_output_shapes
:?????????*
dtype0*
shape:?????????
s
serving_default_video_idPlaceholder*#
_output_shapes
:?????????*
dtype0	*
shape:?????????
w
serving_default_video_ratingPlaceholder*#
_output_shapes
:?????????*
dtype0*
shape:?????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_user_id!serving_default_video_descriptionserving_default_video_idserving_default_video_ratinghash_table_2Const_8embedding/embeddingshash_table_1Const_7embedding_1/embeddings
hash_tableConst_6embedding_2/embeddingsdense/kernel
dense/biasdense_1/kerneldense_1/biasdense_2/kerneldense_2/bias*
Tin
2					*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*+
_read_only_resource_inputs
		*-
config_proto

CPU

GPU 2J 8? *,
f'R%
#__inference_signature_wrapper_15159
?
StatefulPartitionedCall_1StatefulPartitionedCallhash_table_2Const_5Const_4*
Tin
2		*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *'
f"R 
__inference__initializer_15877
?
StatefulPartitionedCall_2StatefulPartitionedCallhash_table_1Const_3Const_2*
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *'
f"R 
__inference__initializer_15895
?
StatefulPartitionedCall_3StatefulPartitionedCall
hash_tableConstConst_1*
Tin
2		*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *'
f"R 
__inference__initializer_15913
`
NoOpNoOp^StatefulPartitionedCall_1^StatefulPartitionedCall_2^StatefulPartitionedCall_3
?g
Const_9Const"/device:CPU:0*
_output_shapes
: *
dtype0*?g
value?gB?g B?g
?
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
ranking_model
	task

	optimizer
loss

signatures*
C
0
1
2
3
4
5
6
7
8*
C
0
1
2
3
4
5
6
7
8*
* 
?
non_trainable_variables

layers
metrics
layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
trace_0
trace_1
trace_2
trace_3* 
6
trace_0
 trace_1
!trace_2
"trace_3* 
/
#	capture_1
$	capture_4
%	capture_7* 
?
&	variables
'trainable_variables
(regularization_losses
)	keras_api
*__call__
*+&call_and_return_all_conditional_losses
,user_embeddings
 -video_description_embeddings
.video_id_embeddings
/ratings*
?
0	variables
1trainable_variables
2regularization_losses
3	keras_api
4__call__
*5&call_and_return_all_conditional_losses
6_ranking_metrics
7_prediction_metrics
8_label_metrics
9_loss_metrics*
?
:iter
	;decay
<learning_rateaccumulator?accumulator?accumulator?accumulator?accumulator?accumulator?accumulator?accumulator?*
* 

=serving_default* 
TN
VARIABLE_VALUEembedding/embeddings&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
VP
VARIABLE_VALUEembedding_1/embeddings&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
VP
VARIABLE_VALUEembedding_2/embeddings&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
LF
VARIABLE_VALUEdense/kernel&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
JD
VARIABLE_VALUE
dense/bias&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUEdense_1/kernel&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
LF
VARIABLE_VALUEdense_1/bias&variables/6/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUEdense_2/kernel&variables/7/.ATTRIBUTES/VARIABLE_VALUE*
LF
VARIABLE_VALUEdense_2/bias&variables/8/.ATTRIBUTES/VARIABLE_VALUE*
* 

0
	1*

>0*
* 
* 
/
#	capture_1
$	capture_4
%	capture_7* 
/
#	capture_1
$	capture_4
%	capture_7* 
/
#	capture_1
$	capture_4
%	capture_7* 
/
#	capture_1
$	capture_4
%	capture_7* 
/
#	capture_1
$	capture_4
%	capture_7* 
/
#	capture_1
$	capture_4
%	capture_7* 
/
#	capture_1
$	capture_4
%	capture_7* 
/
#	capture_1
$	capture_4
%	capture_7* 
* 
* 
* 
C
0
1
2
3
4
5
6
7
8*
C
0
1
2
3
4
5
6
7
8*
* 
?
?non_trainable_variables

@layers
Ametrics
Blayer_regularization_losses
Clayer_metrics
&	variables
'trainable_variables
(regularization_losses
*__call__
*+&call_and_return_all_conditional_losses
&+"call_and_return_conditional_losses*
6
Dtrace_0
Etrace_1
Ftrace_2
Gtrace_3* 
6
Htrace_0
Itrace_1
Jtrace_2
Ktrace_3* 
?
Llayer-0
Mlayer_with_weights-0
Mlayer-1
N	variables
Otrainable_variables
Pregularization_losses
Q	keras_api
R__call__
*S&call_and_return_all_conditional_losses*
?
Tlayer-0
Ulayer_with_weights-0
Ulayer-1
V	variables
Wtrainable_variables
Xregularization_losses
Y	keras_api
Z__call__
*[&call_and_return_all_conditional_losses*
?
\layer-0
]layer_with_weights-0
]layer-1
^	variables
_trainable_variables
`regularization_losses
a	keras_api
b__call__
*c&call_and_return_all_conditional_losses*
?
dlayer_with_weights-0
dlayer-0
elayer_with_weights-1
elayer-1
flayer_with_weights-2
flayer-2
g	variables
htrainable_variables
iregularization_losses
j	keras_api
k__call__
*l&call_and_return_all_conditional_losses*
* 
* 
* 
?
mnon_trainable_variables

nlayers
ometrics
player_regularization_losses
qlayer_metrics
0	variables
1trainable_variables
2regularization_losses
4__call__
*5&call_and_return_all_conditional_losses
&5"call_and_return_conditional_losses*
* 
* 

>0*
* 
* 
* 
OI
VARIABLE_VALUEAdagrad/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
QK
VARIABLE_VALUEAdagrad/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdagrad/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
/
#	capture_1
$	capture_4
%	capture_7* 
8
r	variables
s	keras_api
	ttotal
	ucount*
* 
 
,0
-1
.2
/3*
* 
* 
* 
/
#	capture_1
$	capture_4
%	capture_7* 
/
#	capture_1
$	capture_4
%	capture_7* 
/
#	capture_1
$	capture_4
%	capture_7* 
/
#	capture_1
$	capture_4
%	capture_7* 
/
#	capture_1
$	capture_4
%	capture_7* 
/
#	capture_1
$	capture_4
%	capture_7* 
/
#	capture_1
$	capture_4
%	capture_7* 
/
#	capture_1
$	capture_4
%	capture_7* 
#
v	keras_api
wlookup_table* 
?
x	variables
ytrainable_variables
zregularization_losses
{	keras_api
|__call__
*}&call_and_return_all_conditional_losses

embeddings*

0*

0*
* 
?
~non_trainable_variables

layers
?metrics
 ?layer_regularization_losses
?layer_metrics
N	variables
Otrainable_variables
Pregularization_losses
R__call__
*S&call_and_return_all_conditional_losses
&S"call_and_return_conditional_losses*
:
?trace_0
?trace_1
?trace_2
?trace_3* 
:
?trace_0
?trace_1
?trace_2
?trace_3* 
%
?	keras_api
?lookup_table* 
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses

embeddings*

0*

0*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
V	variables
Wtrainable_variables
Xregularization_losses
Z__call__
*[&call_and_return_all_conditional_losses
&["call_and_return_conditional_losses*
:
?trace_0
?trace_1
?trace_2
?trace_3* 
:
?trace_0
?trace_1
?trace_2
?trace_3* 
%
?	keras_api
?lookup_table* 
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses

embeddings*

0*

0*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
^	variables
_trainable_variables
`regularization_losses
b__call__
*c&call_and_return_all_conditional_losses
&c"call_and_return_conditional_losses*
:
?trace_0
?trace_1
?trace_2
?trace_3* 
:
?trace_0
?trace_1
?trace_2
?trace_3* 
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses

kernel
bias*
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses

kernel
bias*
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses

kernel
bias*
.
0
1
2
3
4
5*
.
0
1
2
3
4
5*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
g	variables
htrainable_variables
iregularization_losses
k__call__
*l&call_and_return_all_conditional_losses
&l"call_and_return_conditional_losses*
:
?trace_0
?trace_1
?trace_2
?trace_3* 
:
?trace_0
?trace_1
?trace_2
?trace_3* 
* 
* 

>0*
* 
!
>root_mean_squared_error*

t0
u1*

r	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*
* 
V
?_initializer
?_create_resource
?_initialize
?_destroy_resource* 

0*

0*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
x	variables
ytrainable_variables
zregularization_losses
|__call__
*}&call_and_return_all_conditional_losses
&}"call_and_return_conditional_losses*

?trace_0* 

?trace_0* 
* 

L0
M1*
* 
* 
* 

#	capture_1* 

#	capture_1* 

#	capture_1* 

#	capture_1* 

#	capture_1* 

#	capture_1* 

#	capture_1* 

#	capture_1* 
* 
V
?_initializer
?_create_resource
?_initialize
?_destroy_resource* 

0*

0*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*

?trace_0* 

?trace_0* 
* 

T0
U1*
* 
* 
* 

$	capture_1* 

$	capture_1* 

$	capture_1* 

$	capture_1* 

$	capture_1* 

$	capture_1* 

$	capture_1* 

$	capture_1* 
* 
V
?_initializer
?_create_resource
?_initialize
?_destroy_resource* 

0*

0*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*

?trace_0* 

?trace_0* 
* 

\0
]1*
* 
* 
* 

%	capture_1* 

%	capture_1* 

%	capture_1* 

%	capture_1* 

%	capture_1* 

%	capture_1* 

%	capture_1* 

%	capture_1* 

0
1*

0
1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*

?trace_0* 

?trace_0* 

0
1*

0
1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*

?trace_0* 

?trace_0* 

0
1*

0
1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*

?trace_0* 

?trace_0* 
* 

d0
e1
f2*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

?trace_0* 

?trace_0* 

?trace_0* 
* 
* 
* 
* 
* 
* 
* 
* 

?trace_0* 

?trace_0* 

?trace_0* 
* 
* 
* 
* 
* 
* 
* 
* 

?trace_0* 

?trace_0* 

?trace_0* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
"
?	capture_1
?	capture_2* 
* 
* 
"
?	capture_1
?	capture_2* 
* 
* 
"
?	capture_1
?	capture_2* 
* 
* 
* 
* 
* 
* 
* 
??
VARIABLE_VALUE(Adagrad/embedding/embeddings/accumulatorLvariables/0/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE*Adagrad/embedding_1/embeddings/accumulatorLvariables/1/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE Adagrad/dense/kernel/accumulatorLvariables/3/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUE*
?~
VARIABLE_VALUEAdagrad/dense/bias/accumulatorLvariables/4/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE"Adagrad/dense_1/kernel/accumulatorLvariables/5/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE Adagrad/dense_1/bias/accumulatorLvariables/6/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE"Adagrad/dense_2/kernel/accumulatorLvariables/7/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE Adagrad/dense_2/bias/accumulatorLvariables/8/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUE*
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?	
StatefulPartitionedCall_4StatefulPartitionedCallsaver_filename(embedding/embeddings/Read/ReadVariableOp*embedding_1/embeddings/Read/ReadVariableOp*embedding_2/embeddings/Read/ReadVariableOp dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOp"dense_2/kernel/Read/ReadVariableOp dense_2/bias/Read/ReadVariableOp Adagrad/iter/Read/ReadVariableOp!Adagrad/decay/Read/ReadVariableOp)Adagrad/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp<Adagrad/embedding/embeddings/accumulator/Read/ReadVariableOp>Adagrad/embedding_1/embeddings/accumulator/Read/ReadVariableOp4Adagrad/dense/kernel/accumulator/Read/ReadVariableOp2Adagrad/dense/bias/accumulator/Read/ReadVariableOp6Adagrad/dense_1/kernel/accumulator/Read/ReadVariableOp4Adagrad/dense_1/bias/accumulator/Read/ReadVariableOp6Adagrad/dense_2/kernel/accumulator/Read/ReadVariableOp4Adagrad/dense_2/bias/accumulator/Read/ReadVariableOpConst_9*#
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *'
f"R 
__inference__traced_save_16025
?
StatefulPartitionedCall_5StatefulPartitionedCallsaver_filenameembedding/embeddingsembedding_1/embeddingsembedding_2/embeddingsdense/kernel
dense/biasdense_1/kerneldense_1/biasdense_2/kerneldense_2/biasAdagrad/iterAdagrad/decayAdagrad/learning_ratetotalcount(Adagrad/embedding/embeddings/accumulator*Adagrad/embedding_1/embeddings/accumulator Adagrad/dense/kernel/accumulatorAdagrad/dense/bias/accumulator"Adagrad/dense_1/kernel/accumulator Adagrad/dense_1/bias/accumulator"Adagrad/dense_2/kernel/accumulator Adagrad/dense_2/bias/accumulator*"
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? **
f%R#
!__inference__traced_restore_16101??
?
?
__inference__initializer_158775
1key_value_init59_lookuptableimportv2_table_handle-
)key_value_init59_lookuptableimportv2_keys	/
+key_value_init59_lookuptableimportv2_values	
identity??$key_value_init59/LookupTableImportV2?
$key_value_init59/LookupTableImportV2LookupTableImportV21key_value_init59_lookuptableimportv2_table_handle)key_value_init59_lookuptableimportv2_keys+key_value_init59_lookuptableimportv2_values*	
Tin0	*

Tout0	*
_output_shapes
 G
ConstConst*
_output_shapes
: *
dtype0*
value	B :L
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: m
NoOpNoOp%^key_value_init59/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
: :#:#2L
$key_value_init59/LookupTableImportV2$key_value_init59/LookupTableImportV2: 

_output_shapes
:#: 

_output_shapes
:#
?
?
G__inference_sequential_1_layer_call_and_return_conditional_losses_14091

inputs<
8string_lookup_none_lookup_lookuptablefindv2_table_handle=
9string_lookup_none_lookup_lookuptablefindv2_default_value	#
embedding_1_14087: 
identity??#embedding_1/StatefulPartitionedCall?+string_lookup/None_Lookup/LookupTableFindV2?
+string_lookup/None_Lookup/LookupTableFindV2LookupTableFindV28string_lookup_none_lookup_lookuptablefindv2_table_handleinputs9string_lookup_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
string_lookup/IdentityIdentity4string_lookup/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:??????????
#embedding_1/StatefulPartitionedCallStatefulPartitionedCallstring_lookup/Identity:output:0embedding_1_14087*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_embedding_1_layer_call_and_return_conditional_losses_14086{
IdentityIdentity,embedding_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:????????? ?
NoOpNoOp$^embedding_1/StatefulPartitionedCall,^string_lookup/None_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????: : : 2J
#embedding_1/StatefulPartitionedCall#embedding_1/StatefulPartitionedCall2Z
+string_lookup/None_Lookup/LookupTableFindV2+string_lookup/None_Lookup/LookupTableFindV2:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: 
?
?
H__inference_ranking_model_layer_call_and_return_conditional_losses_14811
input_1	
input_2	
input_3
sequential_14774
sequential_14776	"
sequential_14778:$ 
sequential_1_14781
sequential_1_14783	$
sequential_1_14785: 
sequential_2_14788
sequential_2_14790	%
sequential_2_14792:	? %
sequential_3_14797:	@?!
sequential_3_14799:	?%
sequential_3_14801:	?@ 
sequential_3_14803:@$
sequential_3_14805:@ 
sequential_3_14807:
identity??"sequential/StatefulPartitionedCall?$sequential_1/StatefulPartitionedCall?$sequential_2/StatefulPartitionedCall?$sequential_3/StatefulPartitionedCall?
"sequential/StatefulPartitionedCallStatefulPartitionedCallinput_1sequential_14774sequential_14776sequential_14778*
Tin
2		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_14024?
$sequential_1/StatefulPartitionedCallStatefulPartitionedCallinput_3sequential_1_14781sequential_1_14783sequential_1_14785*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_sequential_1_layer_call_and_return_conditional_losses_14132?
$sequential_2/StatefulPartitionedCallStatefulPartitionedCallinput_2sequential_2_14788sequential_2_14790sequential_2_14792*
Tin
2		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_sequential_2_layer_call_and_return_conditional_losses_14240M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :?
concatConcatV2+sequential/StatefulPartitionedCall:output:0-sequential_1/StatefulPartitionedCall:output:0concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????@?
$sequential_3/StatefulPartitionedCallStatefulPartitionedCallconcat:output:0sequential_3_14797sequential_3_14799sequential_3_14801sequential_3_14803sequential_3_14805sequential_3_14807*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_sequential_3_layer_call_and_return_conditional_losses_14423|
IdentityIdentity-sequential_3/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp#^sequential/StatefulPartitionedCall%^sequential_1/StatefulPartitionedCall%^sequential_2/StatefulPartitionedCall%^sequential_3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*^
_input_shapesM
K:?????????:?????????:?????????: : : : : : : : : : : : : : : 2H
"sequential/StatefulPartitionedCall"sequential/StatefulPartitionedCall2L
$sequential_1/StatefulPartitionedCall$sequential_1/StatefulPartitionedCall2L
$sequential_2/StatefulPartitionedCall$sequential_2/StatefulPartitionedCall2L
$sequential_3/StatefulPartitionedCall$sequential_3/StatefulPartitionedCall:L H
#
_output_shapes
:?????????
!
_user_specified_name	input_1:LH
#
_output_shapes
:?????????
!
_user_specified_name	input_2:LH
#
_output_shapes
:?????????
!
_user_specified_name	input_3:

_output_shapes
: :

_output_shapes
: :


_output_shapes
: 
?
?
%__inference_dense_layer_call_fn_15814

inputs
unknown:	@?
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_14300p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????@: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
G__inference_sequential_3_layer_call_and_return_conditional_losses_14474
dense_input
dense_14458:	@?
dense_14460:	? 
dense_1_14463:	?@
dense_1_14465:@
dense_2_14468:@
dense_2_14470:
identity??dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?dense_2/StatefulPartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCalldense_inputdense_14458dense_14460*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_14300?
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_14463dense_1_14465*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_14317?
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_14468dense_2_14470*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_14333w
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????@: : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall:T P
'
_output_shapes
:?????????@
%
_user_specified_namedense_input
?
?
+__inference_video_model_layer_call_fn_14889
user_id	
video_description
video_id	
video_rating
unknown
	unknown_0	
	unknown_1:$ 
	unknown_2
	unknown_3	
	unknown_4: 
	unknown_5
	unknown_6	
	unknown_7:	? 
	unknown_8:	@?
	unknown_9:	?

unknown_10:	?@

unknown_11:@

unknown_12:@

unknown_13:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalluser_idvideo_descriptionvideo_idvideo_ratingunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13*
Tin
2					*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*+
_read_only_resource_inputs
		*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_video_model_layer_call_and_return_conditional_losses_14856o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z:?????????:?????????:?????????:?????????: : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:L H
#
_output_shapes
:?????????
!
_user_specified_name	user_id:VR
#
_output_shapes
:?????????
+
_user_specified_namevideo_description:MI
#
_output_shapes
:?????????
"
_user_specified_name
video_id:QM
#
_output_shapes
:?????????
&
_user_specified_namevideo_rating:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
+__inference_video_model_layer_call_fn_15235
features_user_id	
features_video_description
features_video_id	
features_video_rating
unknown
	unknown_0	
	unknown_1:$ 
	unknown_2
	unknown_3	
	unknown_4: 
	unknown_5
	unknown_6	
	unknown_7:	? 
	unknown_8:	@?
	unknown_9:	?

unknown_10:	?@

unknown_11:@

unknown_12:@

unknown_13:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallfeatures_user_idfeatures_video_descriptionfeatures_video_idfeatures_video_ratingunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13*
Tin
2					*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*+
_read_only_resource_inputs
		*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_video_model_layer_call_and_return_conditional_losses_14970o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z:?????????:?????????:?????????:?????????: : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
#
_output_shapes
:?????????
*
_user_specified_namefeatures/user_id:_[
#
_output_shapes
:?????????
4
_user_specified_namefeatures/video_description:VR
#
_output_shapes
:?????????
+
_user_specified_namefeatures/video_id:ZV
#
_output_shapes
:?????????
/
_user_specified_namefeatures/video_rating:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
,__inference_sequential_1_layer_call_fn_14100
string_lookup_input
unknown
	unknown_0	
	unknown_1: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallstring_lookup_inputunknown	unknown_0	unknown_1*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_sequential_1_layer_call_and_return_conditional_losses_14091o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:????????? `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
#
_output_shapes
:?????????
-
_user_specified_namestring_lookup_input:

_output_shapes
: 
?
?
F__inference_embedding_2_layer_call_and_return_conditional_losses_15805

inputs	)
embedding_lookup_15799:	? 
identity??embedding_lookup?
embedding_lookupResourceGatherembedding_lookup_15799inputs*
Tindices0	*)
_class
loc:@embedding_lookup/15799*'
_output_shapes
:????????? *
dtype0?
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*)
_class
loc:@embedding_lookup/15799*'
_output_shapes
:????????? }
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:????????? s
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*'
_output_shapes
:????????? Y
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:?????????: 2$
embedding_lookupembedding_lookup:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
G__inference_sequential_1_layer_call_and_return_conditional_losses_15614

inputs<
8string_lookup_none_lookup_lookuptablefindv2_table_handle=
9string_lookup_none_lookup_lookuptablefindv2_default_value	4
"embedding_1_embedding_lookup_15608: 
identity??embedding_1/embedding_lookup?+string_lookup/None_Lookup/LookupTableFindV2?
+string_lookup/None_Lookup/LookupTableFindV2LookupTableFindV28string_lookup_none_lookup_lookuptablefindv2_table_handleinputs9string_lookup_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
string_lookup/IdentityIdentity4string_lookup/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:??????????
embedding_1/embedding_lookupResourceGather"embedding_1_embedding_lookup_15608string_lookup/Identity:output:0*
Tindices0	*5
_class+
)'loc:@embedding_1/embedding_lookup/15608*'
_output_shapes
:????????? *
dtype0?
%embedding_1/embedding_lookup/IdentityIdentity%embedding_1/embedding_lookup:output:0*
T0*5
_class+
)'loc:@embedding_1/embedding_lookup/15608*'
_output_shapes
:????????? ?
'embedding_1/embedding_lookup/Identity_1Identity.embedding_1/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:????????? 
IdentityIdentity0embedding_1/embedding_lookup/Identity_1:output:0^NoOp*
T0*'
_output_shapes
:????????? ?
NoOpNoOp^embedding_1/embedding_lookup,^string_lookup/None_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????: : : 2<
embedding_1/embedding_lookupembedding_1/embedding_lookup2Z
+string_lookup/None_Lookup/LookupTableFindV2+string_lookup/None_Lookup/LookupTableFindV2:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: 
?
?
,__inference_sequential_2_layer_call_fn_14208
integer_lookup_1_input	
unknown
	unknown_0	
	unknown_1:	? 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinteger_lookup_1_inputunknown	unknown_0	unknown_1*
Tin
2		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_sequential_2_layer_call_and_return_conditional_losses_14199o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:????????? `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
#
_output_shapes
:?????????
0
_user_specified_nameinteger_lookup_1_input:

_output_shapes
: 
?
?
H__inference_ranking_model_layer_call_and_return_conditional_losses_14769
input_1	
input_2	
input_3
sequential_14732
sequential_14734	"
sequential_14736:$ 
sequential_1_14739
sequential_1_14741	$
sequential_1_14743: 
sequential_2_14746
sequential_2_14748	%
sequential_2_14750:	? %
sequential_3_14755:	@?!
sequential_3_14757:	?%
sequential_3_14759:	?@ 
sequential_3_14761:@$
sequential_3_14763:@ 
sequential_3_14765:
identity??"sequential/StatefulPartitionedCall?$sequential_1/StatefulPartitionedCall?$sequential_2/StatefulPartitionedCall?$sequential_3/StatefulPartitionedCall?
"sequential/StatefulPartitionedCallStatefulPartitionedCallinput_1sequential_14732sequential_14734sequential_14736*
Tin
2		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_13983?
$sequential_1/StatefulPartitionedCallStatefulPartitionedCallinput_3sequential_1_14739sequential_1_14741sequential_1_14743*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_sequential_1_layer_call_and_return_conditional_losses_14091?
$sequential_2/StatefulPartitionedCallStatefulPartitionedCallinput_2sequential_2_14746sequential_2_14748sequential_2_14750*
Tin
2		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_sequential_2_layer_call_and_return_conditional_losses_14199M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :?
concatConcatV2+sequential/StatefulPartitionedCall:output:0-sequential_1/StatefulPartitionedCall:output:0concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????@?
$sequential_3/StatefulPartitionedCallStatefulPartitionedCallconcat:output:0sequential_3_14755sequential_3_14757sequential_3_14759sequential_3_14761sequential_3_14763sequential_3_14765*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_sequential_3_layer_call_and_return_conditional_losses_14340|
IdentityIdentity-sequential_3/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp#^sequential/StatefulPartitionedCall%^sequential_1/StatefulPartitionedCall%^sequential_2/StatefulPartitionedCall%^sequential_3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*^
_input_shapesM
K:?????????:?????????:?????????: : : : : : : : : : : : : : : 2H
"sequential/StatefulPartitionedCall"sequential/StatefulPartitionedCall2L
$sequential_1/StatefulPartitionedCall$sequential_1/StatefulPartitionedCall2L
$sequential_2/StatefulPartitionedCall$sequential_2/StatefulPartitionedCall2L
$sequential_3/StatefulPartitionedCall$sequential_3/StatefulPartitionedCall:L H
#
_output_shapes
:?????????
!
_user_specified_name	input_1:LH
#
_output_shapes
:?????????
!
_user_specified_name	input_2:LH
#
_output_shapes
:?????????
!
_user_specified_name	input_3:

_output_shapes
: :

_output_shapes
: :


_output_shapes
: 
?
?
E__inference_sequential_layer_call_and_return_conditional_losses_15579

inputs	=
9integer_lookup_none_lookup_lookuptablefindv2_table_handle>
:integer_lookup_none_lookup_lookuptablefindv2_default_value	2
 embedding_embedding_lookup_15573:$ 
identity??embedding/embedding_lookup?,integer_lookup/None_Lookup/LookupTableFindV2?
,integer_lookup/None_Lookup/LookupTableFindV2LookupTableFindV29integer_lookup_none_lookup_lookuptablefindv2_table_handleinputs:integer_lookup_none_lookup_lookuptablefindv2_default_value*	
Tin0	*

Tout0	*#
_output_shapes
:??????????
integer_lookup/IdentityIdentity5integer_lookup/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:??????????
embedding/embedding_lookupResourceGather embedding_embedding_lookup_15573 integer_lookup/Identity:output:0*
Tindices0	*3
_class)
'%loc:@embedding/embedding_lookup/15573*'
_output_shapes
:????????? *
dtype0?
#embedding/embedding_lookup/IdentityIdentity#embedding/embedding_lookup:output:0*
T0*3
_class)
'%loc:@embedding/embedding_lookup/15573*'
_output_shapes
:????????? ?
%embedding/embedding_lookup/Identity_1Identity,embedding/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:????????? }
IdentityIdentity.embedding/embedding_lookup/Identity_1:output:0^NoOp*
T0*'
_output_shapes
:????????? ?
NoOpNoOp^embedding/embedding_lookup-^integer_lookup/None_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????: : : 28
embedding/embedding_lookupembedding/embedding_lookup2\
,integer_lookup/None_Lookup/LookupTableFindV2,integer_lookup/None_Lookup/LookupTableFindV2:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: 
?
?
*__inference_sequential_layer_call_fn_14044
integer_lookup_input	
unknown
	unknown_0	
	unknown_1:$ 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinteger_lookup_inputunknown	unknown_0	unknown_1*
Tin
2		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_14024o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:????????? `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
#
_output_shapes
:?????????
.
_user_specified_nameinteger_lookup_input:

_output_shapes
: 
?
?
#__inference_signature_wrapper_15159
user_id	
video_description
video_id	
video_rating
unknown
	unknown_0	
	unknown_1:$ 
	unknown_2
	unknown_3	
	unknown_4: 
	unknown_5
	unknown_6	
	unknown_7:	? 
	unknown_8:	@?
	unknown_9:	?

unknown_10:	?@

unknown_11:@

unknown_12:@

unknown_13:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalluser_idvideo_descriptionvideo_idvideo_ratingunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13*
Tin
2					*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*+
_read_only_resource_inputs
		*-
config_proto

CPU

GPU 2J 8? *)
f$R"
 __inference__wrapped_model_13958o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z:?????????:?????????:?????????:?????????: : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:L H
#
_output_shapes
:?????????
!
_user_specified_name	user_id:VR
#
_output_shapes
:?????????
+
_user_specified_namevideo_description:MI
#
_output_shapes
:?????????
"
_user_specified_name
video_id:QM
#
_output_shapes
:?????????
&
_user_specified_namevideo_rating:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
-__inference_ranking_model_layer_call_fn_14727
input_1	
input_2	
input_3
unknown
	unknown_0	
	unknown_1:$ 
	unknown_2
	unknown_3	
	unknown_4: 
	unknown_5
	unknown_6	
	unknown_7:	? 
	unknown_8:	@?
	unknown_9:	?

unknown_10:	?@

unknown_11:@

unknown_12:@

unknown_13:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1input_2input_3unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13*
Tin
2					*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*+
_read_only_resource_inputs
	*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_ranking_model_layer_call_and_return_conditional_losses_14657o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*^
_input_shapesM
K:?????????:?????????:?????????: : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:L H
#
_output_shapes
:?????????
!
_user_specified_name	input_1:LH
#
_output_shapes
:?????????
!
_user_specified_name	input_2:LH
#
_output_shapes
:?????????
!
_user_specified_name	input_3:

_output_shapes
: :

_output_shapes
: :


_output_shapes
: 
?
?
G__inference_sequential_3_layer_call_and_return_conditional_losses_15733

inputs7
$dense_matmul_readvariableop_resource:	@?4
%dense_biasadd_readvariableop_resource:	?9
&dense_1_matmul_readvariableop_resource:	?@5
'dense_1_biasadd_readvariableop_resource:@8
&dense_2_matmul_readvariableop_resource:@5
'dense_2_biasadd_readvariableop_resource:
identity??dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?dense_2/BiasAdd/ReadVariableOp?dense_2/MatMul/ReadVariableOp?
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	@?*
dtype0v
dense/MatMulMatMulinputs#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????]

dense/ReluReludense/BiasAdd:output:0*
T0*(
_output_shapes
:???????????
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	?@*
dtype0?
dense_1/MatMulMatMuldense/Relu:activations:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@`
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@?
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0?
dense_2/MatMulMatMuldense_1/Relu:activations:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????g
IdentityIdentitydense_2/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????@: : : : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?

?
@__inference_dense_layer_call_and_return_conditional_losses_15825

inputs1
matmul_readvariableop_resource:	@?.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	@?*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:??????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
G__inference_sequential_2_layer_call_and_return_conditional_losses_14199

inputs	?
;integer_lookup_1_none_lookup_lookuptablefindv2_table_handle@
<integer_lookup_1_none_lookup_lookuptablefindv2_default_value	$
embedding_2_14195:	? 
identity??#embedding_2/StatefulPartitionedCall?.integer_lookup_1/None_Lookup/LookupTableFindV2?
.integer_lookup_1/None_Lookup/LookupTableFindV2LookupTableFindV2;integer_lookup_1_none_lookup_lookuptablefindv2_table_handleinputs<integer_lookup_1_none_lookup_lookuptablefindv2_default_value*	
Tin0	*

Tout0	*#
_output_shapes
:??????????
integer_lookup_1/IdentityIdentity7integer_lookup_1/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:??????????
#embedding_2/StatefulPartitionedCallStatefulPartitionedCall"integer_lookup_1/Identity:output:0embedding_2_14195*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_embedding_2_layer_call_and_return_conditional_losses_14194{
IdentityIdentity,embedding_2/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:????????? ?
NoOpNoOp$^embedding_2/StatefulPartitionedCall/^integer_lookup_1/None_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????: : : 2J
#embedding_2/StatefulPartitionedCall#embedding_2/StatefulPartitionedCall2`
.integer_lookup_1/None_Lookup/LookupTableFindV2.integer_lookup_1/None_Lookup/LookupTableFindV2:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: 
?
?
E__inference_sequential_layer_call_and_return_conditional_losses_14055
integer_lookup_input	=
9integer_lookup_none_lookup_lookuptablefindv2_table_handle>
:integer_lookup_none_lookup_lookuptablefindv2_default_value	!
embedding_14051:$ 
identity??!embedding/StatefulPartitionedCall?,integer_lookup/None_Lookup/LookupTableFindV2?
,integer_lookup/None_Lookup/LookupTableFindV2LookupTableFindV29integer_lookup_none_lookup_lookuptablefindv2_table_handleinteger_lookup_input:integer_lookup_none_lookup_lookuptablefindv2_default_value*	
Tin0	*

Tout0	*#
_output_shapes
:??????????
integer_lookup/IdentityIdentity5integer_lookup/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:??????????
!embedding/StatefulPartitionedCallStatefulPartitionedCall integer_lookup/Identity:output:0embedding_14051*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_embedding_layer_call_and_return_conditional_losses_13978y
IdentityIdentity*embedding/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:????????? ?
NoOpNoOp"^embedding/StatefulPartitionedCall-^integer_lookup/None_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????: : : 2F
!embedding/StatefulPartitionedCall!embedding/StatefulPartitionedCall2\
,integer_lookup/None_Lookup/LookupTableFindV2,integer_lookup/None_Lookup/LookupTableFindV2:Y U
#
_output_shapes
:?????????
.
_user_specified_nameinteger_lookup_input:

_output_shapes
: 
?
?
G__inference_sequential_3_layer_call_and_return_conditional_losses_15757

inputs7
$dense_matmul_readvariableop_resource:	@?4
%dense_biasadd_readvariableop_resource:	?9
&dense_1_matmul_readvariableop_resource:	?@5
'dense_1_biasadd_readvariableop_resource:@8
&dense_2_matmul_readvariableop_resource:@5
'dense_2_biasadd_readvariableop_resource:
identity??dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?dense_2/BiasAdd/ReadVariableOp?dense_2/MatMul/ReadVariableOp?
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	@?*
dtype0v
dense/MatMulMatMulinputs#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????]

dense/ReluReludense/BiasAdd:output:0*
T0*(
_output_shapes
:???????????
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	?@*
dtype0?
dense_1/MatMulMatMuldense/Relu:activations:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@`
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@?
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0?
dense_2/MatMulMatMuldense_1/Relu:activations:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????g
IdentityIdentitydense_2/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????@: : : : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?

?
B__inference_dense_1_layer_call_and_return_conditional_losses_15845

inputs1
matmul_readvariableop_resource:	?@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????@a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????@w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
,
__inference__destroyer_15900
identityG
ConstConst*
_output_shapes
: *
dtype0*
value	B :E
IdentityIdentityConst:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
?
?
D__inference_embedding_layer_call_and_return_conditional_losses_15773

inputs	(
embedding_lookup_15767:$ 
identity??embedding_lookup?
embedding_lookupResourceGatherembedding_lookup_15767inputs*
Tindices0	*)
_class
loc:@embedding_lookup/15767*'
_output_shapes
:????????? *
dtype0?
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*)
_class
loc:@embedding_lookup/15767*'
_output_shapes
:????????? }
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:????????? s
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*'
_output_shapes
:????????? Y
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:?????????: 2$
embedding_lookupembedding_lookup:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs
?
,
__inference__destroyer_15918
identityG
ConstConst*
_output_shapes
: *
dtype0*
value	B :E
IdentityIdentityConst:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
?
?
-__inference_ranking_model_layer_call_fn_14574
input_1	
input_2	
input_3
unknown
	unknown_0	
	unknown_1:$ 
	unknown_2
	unknown_3	
	unknown_4: 
	unknown_5
	unknown_6	
	unknown_7:	? 
	unknown_8:	@?
	unknown_9:	?

unknown_10:	?@

unknown_11:@

unknown_12:@

unknown_13:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1input_2input_3unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13*
Tin
2					*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*+
_read_only_resource_inputs
	*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_ranking_model_layer_call_and_return_conditional_losses_14541o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*^
_input_shapesM
K:?????????:?????????:?????????: : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:L H
#
_output_shapes
:?????????
!
_user_specified_name	input_1:LH
#
_output_shapes
:?????????
!
_user_specified_name	input_2:LH
#
_output_shapes
:?????????
!
_user_specified_name	input_3:

_output_shapes
: :

_output_shapes
: :


_output_shapes
: 
?
?
+__inference_embedding_2_layer_call_fn_15796

inputs	
unknown:	? 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_embedding_2_layer_call_and_return_conditional_losses_14194o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:????????? `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:?????????: 22
StatefulPartitionedCallStatefulPartitionedCall:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
,__inference_sequential_2_layer_call_fn_15649

inputs	
unknown
	unknown_0	
	unknown_1:	? 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_sequential_2_layer_call_and_return_conditional_losses_14240o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:????????? `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: 
?
?
G__inference_sequential_1_layer_call_and_return_conditional_losses_14174
string_lookup_input<
8string_lookup_none_lookup_lookuptablefindv2_table_handle=
9string_lookup_none_lookup_lookuptablefindv2_default_value	#
embedding_1_14170: 
identity??#embedding_1/StatefulPartitionedCall?+string_lookup/None_Lookup/LookupTableFindV2?
+string_lookup/None_Lookup/LookupTableFindV2LookupTableFindV28string_lookup_none_lookup_lookuptablefindv2_table_handlestring_lookup_input9string_lookup_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
string_lookup/IdentityIdentity4string_lookup/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:??????????
#embedding_1/StatefulPartitionedCallStatefulPartitionedCallstring_lookup/Identity:output:0embedding_1_14170*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_embedding_1_layer_call_and_return_conditional_losses_14086{
IdentityIdentity,embedding_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:????????? ?
NoOpNoOp$^embedding_1/StatefulPartitionedCall,^string_lookup/None_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????: : : 2J
#embedding_1/StatefulPartitionedCall#embedding_1/StatefulPartitionedCall2Z
+string_lookup/None_Lookup/LookupTableFindV2+string_lookup/None_Lookup/LookupTableFindV2:X T
#
_output_shapes
:?????????
-
_user_specified_namestring_lookup_input:

_output_shapes
: 
?
?
G__inference_sequential_2_layer_call_and_return_conditional_losses_14282
integer_lookup_1_input	?
;integer_lookup_1_none_lookup_lookuptablefindv2_table_handle@
<integer_lookup_1_none_lookup_lookuptablefindv2_default_value	$
embedding_2_14278:	? 
identity??#embedding_2/StatefulPartitionedCall?.integer_lookup_1/None_Lookup/LookupTableFindV2?
.integer_lookup_1/None_Lookup/LookupTableFindV2LookupTableFindV2;integer_lookup_1_none_lookup_lookuptablefindv2_table_handleinteger_lookup_1_input<integer_lookup_1_none_lookup_lookuptablefindv2_default_value*	
Tin0	*

Tout0	*#
_output_shapes
:??????????
integer_lookup_1/IdentityIdentity7integer_lookup_1/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:??????????
#embedding_2/StatefulPartitionedCallStatefulPartitionedCall"integer_lookup_1/Identity:output:0embedding_2_14278*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_embedding_2_layer_call_and_return_conditional_losses_14194{
IdentityIdentity,embedding_2/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:????????? ?
NoOpNoOp$^embedding_2/StatefulPartitionedCall/^integer_lookup_1/None_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????: : : 2J
#embedding_2/StatefulPartitionedCall#embedding_2/StatefulPartitionedCall2`
.integer_lookup_1/None_Lookup/LookupTableFindV2.integer_lookup_1/None_Lookup/LookupTableFindV2:[ W
#
_output_shapes
:?????????
0
_user_specified_nameinteger_lookup_1_input:

_output_shapes
: 
?	
?
,__inference_sequential_3_layer_call_fn_14455
dense_input
unknown:	@?
	unknown_0:	?
	unknown_1:	?@
	unknown_2:@
	unknown_3:@
	unknown_4:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalldense_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_sequential_3_layer_call_and_return_conditional_losses_14423o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????@: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
'
_output_shapes
:?????????@
%
_user_specified_namedense_input
?

?
@__inference_dense_layer_call_and_return_conditional_losses_14300

inputs1
matmul_readvariableop_resource:	@?.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	@?*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:??????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
G__inference_sequential_2_layer_call_and_return_conditional_losses_14271
integer_lookup_1_input	?
;integer_lookup_1_none_lookup_lookuptablefindv2_table_handle@
<integer_lookup_1_none_lookup_lookuptablefindv2_default_value	$
embedding_2_14267:	? 
identity??#embedding_2/StatefulPartitionedCall?.integer_lookup_1/None_Lookup/LookupTableFindV2?
.integer_lookup_1/None_Lookup/LookupTableFindV2LookupTableFindV2;integer_lookup_1_none_lookup_lookuptablefindv2_table_handleinteger_lookup_1_input<integer_lookup_1_none_lookup_lookuptablefindv2_default_value*	
Tin0	*

Tout0	*#
_output_shapes
:??????????
integer_lookup_1/IdentityIdentity7integer_lookup_1/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:??????????
#embedding_2/StatefulPartitionedCallStatefulPartitionedCall"integer_lookup_1/Identity:output:0embedding_2_14267*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_embedding_2_layer_call_and_return_conditional_losses_14194{
IdentityIdentity,embedding_2/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:????????? ?
NoOpNoOp$^embedding_2/StatefulPartitionedCall/^integer_lookup_1/None_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????: : : 2J
#embedding_2/StatefulPartitionedCall#embedding_2/StatefulPartitionedCall2`
.integer_lookup_1/None_Lookup/LookupTableFindV2.integer_lookup_1/None_Lookup/LookupTableFindV2:[ W
#
_output_shapes
:?????????
0
_user_specified_nameinteger_lookup_1_input:

_output_shapes
: 
?
?
G__inference_sequential_1_layer_call_and_return_conditional_losses_14163
string_lookup_input<
8string_lookup_none_lookup_lookuptablefindv2_table_handle=
9string_lookup_none_lookup_lookuptablefindv2_default_value	#
embedding_1_14159: 
identity??#embedding_1/StatefulPartitionedCall?+string_lookup/None_Lookup/LookupTableFindV2?
+string_lookup/None_Lookup/LookupTableFindV2LookupTableFindV28string_lookup_none_lookup_lookuptablefindv2_table_handlestring_lookup_input9string_lookup_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
string_lookup/IdentityIdentity4string_lookup/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:??????????
#embedding_1/StatefulPartitionedCallStatefulPartitionedCallstring_lookup/Identity:output:0embedding_1_14159*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_embedding_1_layer_call_and_return_conditional_losses_14086{
IdentityIdentity,embedding_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:????????? ?
NoOpNoOp$^embedding_1/StatefulPartitionedCall,^string_lookup/None_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????: : : 2J
#embedding_1/StatefulPartitionedCall#embedding_1/StatefulPartitionedCall2Z
+string_lookup/None_Lookup/LookupTableFindV2+string_lookup/None_Lookup/LookupTableFindV2:X T
#
_output_shapes
:?????????
-
_user_specified_namestring_lookup_input:

_output_shapes
: 
?	
?
B__inference_dense_2_layer_call_and_return_conditional_losses_15864

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
,
__inference__destroyer_15882
identityG
ConstConst*
_output_shapes
: *
dtype0*
value	B :E
IdentityIdentityConst:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
?
?
,__inference_sequential_3_layer_call_fn_15709

inputs
unknown:	@?
	unknown_0:	?
	unknown_1:	?@
	unknown_2:@
	unknown_3:@
	unknown_4:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_sequential_3_layer_call_and_return_conditional_losses_14423o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????@: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
G__inference_sequential_1_layer_call_and_return_conditional_losses_15627

inputs<
8string_lookup_none_lookup_lookuptablefindv2_table_handle=
9string_lookup_none_lookup_lookuptablefindv2_default_value	4
"embedding_1_embedding_lookup_15621: 
identity??embedding_1/embedding_lookup?+string_lookup/None_Lookup/LookupTableFindV2?
+string_lookup/None_Lookup/LookupTableFindV2LookupTableFindV28string_lookup_none_lookup_lookuptablefindv2_table_handleinputs9string_lookup_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
string_lookup/IdentityIdentity4string_lookup/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:??????????
embedding_1/embedding_lookupResourceGather"embedding_1_embedding_lookup_15621string_lookup/Identity:output:0*
Tindices0	*5
_class+
)'loc:@embedding_1/embedding_lookup/15621*'
_output_shapes
:????????? *
dtype0?
%embedding_1/embedding_lookup/IdentityIdentity%embedding_1/embedding_lookup:output:0*
T0*5
_class+
)'loc:@embedding_1/embedding_lookup/15621*'
_output_shapes
:????????? ?
'embedding_1/embedding_lookup/Identity_1Identity.embedding_1/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:????????? 
IdentityIdentity0embedding_1/embedding_lookup/Identity_1:output:0^NoOp*
T0*'
_output_shapes
:????????? ?
NoOpNoOp^embedding_1/embedding_lookup,^string_lookup/None_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????: : : 2<
embedding_1/embedding_lookupembedding_1/embedding_lookup2Z
+string_lookup/None_Lookup/LookupTableFindV2+string_lookup/None_Lookup/LookupTableFindV2:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: 
?

+__inference_embedding_1_layer_call_fn_15780

inputs	
unknown: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_embedding_1_layer_call_and_return_conditional_losses_14086o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:????????? `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:?????????: 22
StatefulPartitionedCallStatefulPartitionedCall:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
G__inference_sequential_3_layer_call_and_return_conditional_losses_14340

inputs
dense_14301:	@?
dense_14303:	? 
dense_1_14318:	?@
dense_1_14320:@
dense_2_14334:@
dense_2_14336:
identity??dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?dense_2/StatefulPartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCallinputsdense_14301dense_14303*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_14300?
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_14318dense_1_14320*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_14317?
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_14334dense_2_14336*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_14333w
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????@: : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?Z
?
!__inference__traced_restore_16101
file_prefix7
%assignvariableop_embedding_embeddings:$ ;
)assignvariableop_1_embedding_1_embeddings: <
)assignvariableop_2_embedding_2_embeddings:	? 2
assignvariableop_3_dense_kernel:	@?,
assignvariableop_4_dense_bias:	?4
!assignvariableop_5_dense_1_kernel:	?@-
assignvariableop_6_dense_1_bias:@3
!assignvariableop_7_dense_2_kernel:@-
assignvariableop_8_dense_2_bias:)
assignvariableop_9_adagrad_iter:	 +
!assignvariableop_10_adagrad_decay: 3
)assignvariableop_11_adagrad_learning_rate: #
assignvariableop_12_total: #
assignvariableop_13_count: N
<assignvariableop_14_adagrad_embedding_embeddings_accumulator:$ P
>assignvariableop_15_adagrad_embedding_1_embeddings_accumulator: G
4assignvariableop_16_adagrad_dense_kernel_accumulator:	@?A
2assignvariableop_17_adagrad_dense_bias_accumulator:	?I
6assignvariableop_18_adagrad_dense_1_kernel_accumulator:	?@B
4assignvariableop_19_adagrad_dense_1_bias_accumulator:@H
6assignvariableop_20_adagrad_dense_2_kernel_accumulator:@B
4assignvariableop_21_adagrad_dense_2_bias_accumulator:
identity_23??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?

RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?

value?	B?	B&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBLvariables/0/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEBLvariables/1/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEBLvariables/3/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEBLvariables/4/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEBLvariables/5/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEBLvariables/6/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEBLvariables/7/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEBLvariables/8/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*A
value8B6B B B B B B B B B B B B B B B B B B B B B B B ?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*p
_output_shapes^
\:::::::::::::::::::::::*%
dtypes
2	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOpAssignVariableOp%assignvariableop_embedding_embeddingsIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_1AssignVariableOp)assignvariableop_1_embedding_1_embeddingsIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_2AssignVariableOp)assignvariableop_2_embedding_2_embeddingsIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_3AssignVariableOpassignvariableop_3_dense_kernelIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_4AssignVariableOpassignvariableop_4_dense_biasIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_5AssignVariableOp!assignvariableop_5_dense_1_kernelIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_6AssignVariableOpassignvariableop_6_dense_1_biasIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_7AssignVariableOp!assignvariableop_7_dense_2_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_8AssignVariableOpassignvariableop_8_dense_2_biasIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0	*
_output_shapes
:?
AssignVariableOp_9AssignVariableOpassignvariableop_9_adagrad_iterIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_10AssignVariableOp!assignvariableop_10_adagrad_decayIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_11AssignVariableOp)assignvariableop_11_adagrad_learning_rateIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_12AssignVariableOpassignvariableop_12_totalIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_13AssignVariableOpassignvariableop_13_countIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_14AssignVariableOp<assignvariableop_14_adagrad_embedding_embeddings_accumulatorIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_15AssignVariableOp>assignvariableop_15_adagrad_embedding_1_embeddings_accumulatorIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_16AssignVariableOp4assignvariableop_16_adagrad_dense_kernel_accumulatorIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_17AssignVariableOp2assignvariableop_17_adagrad_dense_bias_accumulatorIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_18AssignVariableOp6assignvariableop_18_adagrad_dense_1_kernel_accumulatorIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_19AssignVariableOp4assignvariableop_19_adagrad_dense_1_bias_accumulatorIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_20AssignVariableOp6assignvariableop_20_adagrad_dense_2_kernel_accumulatorIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_21AssignVariableOp4assignvariableop_21_adagrad_dense_2_bias_accumulatorIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ?
Identity_22Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_23IdentityIdentity_22:output:0^NoOp_1*
T0*
_output_shapes
: ?
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_23Identity_23:output:0*A
_input_shapes0
.: : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?n
?
 __inference__wrapped_model_13958
user_id	
video_description
video_id	
video_ratingb
^video_model_ranking_model_sequential_integer_lookup_none_lookup_lookuptablefindv2_table_handlec
_video_model_ranking_model_sequential_integer_lookup_none_lookup_lookuptablefindv2_default_value	W
Evideo_model_ranking_model_sequential_embedding_embedding_lookup_13860:$ c
_video_model_ranking_model_sequential_1_string_lookup_none_lookup_lookuptablefindv2_table_handled
`video_model_ranking_model_sequential_1_string_lookup_none_lookup_lookuptablefindv2_default_value	[
Ivideo_model_ranking_model_sequential_1_embedding_1_embedding_lookup_13882: f
bvideo_model_ranking_model_sequential_2_integer_lookup_1_none_lookup_lookuptablefindv2_table_handleg
cvideo_model_ranking_model_sequential_2_integer_lookup_1_none_lookup_lookuptablefindv2_default_value	\
Ivideo_model_ranking_model_sequential_2_embedding_2_embedding_lookup_13904:	? ^
Kvideo_model_ranking_model_sequential_3_dense_matmul_readvariableop_resource:	@?[
Lvideo_model_ranking_model_sequential_3_dense_biasadd_readvariableop_resource:	?`
Mvideo_model_ranking_model_sequential_3_dense_1_matmul_readvariableop_resource:	?@\
Nvideo_model_ranking_model_sequential_3_dense_1_biasadd_readvariableop_resource:@_
Mvideo_model_ranking_model_sequential_3_dense_2_matmul_readvariableop_resource:@\
Nvideo_model_ranking_model_sequential_3_dense_2_biasadd_readvariableop_resource:
identity???video_model/ranking_model/sequential/embedding/embedding_lookup?Qvideo_model/ranking_model/sequential/integer_lookup/None_Lookup/LookupTableFindV2?Cvideo_model/ranking_model/sequential_1/embedding_1/embedding_lookup?Rvideo_model/ranking_model/sequential_1/string_lookup/None_Lookup/LookupTableFindV2?Cvideo_model/ranking_model/sequential_2/embedding_2/embedding_lookup?Uvideo_model/ranking_model/sequential_2/integer_lookup_1/None_Lookup/LookupTableFindV2?Cvideo_model/ranking_model/sequential_3/dense/BiasAdd/ReadVariableOp?Bvideo_model/ranking_model/sequential_3/dense/MatMul/ReadVariableOp?Evideo_model/ranking_model/sequential_3/dense_1/BiasAdd/ReadVariableOp?Dvideo_model/ranking_model/sequential_3/dense_1/MatMul/ReadVariableOp?Evideo_model/ranking_model/sequential_3/dense_2/BiasAdd/ReadVariableOp?Dvideo_model/ranking_model/sequential_3/dense_2/MatMul/ReadVariableOp?
Qvideo_model/ranking_model/sequential/integer_lookup/None_Lookup/LookupTableFindV2LookupTableFindV2^video_model_ranking_model_sequential_integer_lookup_none_lookup_lookuptablefindv2_table_handleuser_id_video_model_ranking_model_sequential_integer_lookup_none_lookup_lookuptablefindv2_default_value*	
Tin0	*

Tout0	*#
_output_shapes
:??????????
<video_model/ranking_model/sequential/integer_lookup/IdentityIdentityZvideo_model/ranking_model/sequential/integer_lookup/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:??????????
?video_model/ranking_model/sequential/embedding/embedding_lookupResourceGatherEvideo_model_ranking_model_sequential_embedding_embedding_lookup_13860Evideo_model/ranking_model/sequential/integer_lookup/Identity:output:0*
Tindices0	*X
_classN
LJloc:@video_model/ranking_model/sequential/embedding/embedding_lookup/13860*'
_output_shapes
:????????? *
dtype0?
Hvideo_model/ranking_model/sequential/embedding/embedding_lookup/IdentityIdentityHvideo_model/ranking_model/sequential/embedding/embedding_lookup:output:0*
T0*X
_classN
LJloc:@video_model/ranking_model/sequential/embedding/embedding_lookup/13860*'
_output_shapes
:????????? ?
Jvideo_model/ranking_model/sequential/embedding/embedding_lookup/Identity_1IdentityQvideo_model/ranking_model/sequential/embedding/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:????????? ?
Rvideo_model/ranking_model/sequential_1/string_lookup/None_Lookup/LookupTableFindV2LookupTableFindV2_video_model_ranking_model_sequential_1_string_lookup_none_lookup_lookuptablefindv2_table_handlevideo_description`video_model_ranking_model_sequential_1_string_lookup_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
=video_model/ranking_model/sequential_1/string_lookup/IdentityIdentity[video_model/ranking_model/sequential_1/string_lookup/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:??????????
Cvideo_model/ranking_model/sequential_1/embedding_1/embedding_lookupResourceGatherIvideo_model_ranking_model_sequential_1_embedding_1_embedding_lookup_13882Fvideo_model/ranking_model/sequential_1/string_lookup/Identity:output:0*
Tindices0	*\
_classR
PNloc:@video_model/ranking_model/sequential_1/embedding_1/embedding_lookup/13882*'
_output_shapes
:????????? *
dtype0?
Lvideo_model/ranking_model/sequential_1/embedding_1/embedding_lookup/IdentityIdentityLvideo_model/ranking_model/sequential_1/embedding_1/embedding_lookup:output:0*
T0*\
_classR
PNloc:@video_model/ranking_model/sequential_1/embedding_1/embedding_lookup/13882*'
_output_shapes
:????????? ?
Nvideo_model/ranking_model/sequential_1/embedding_1/embedding_lookup/Identity_1IdentityUvideo_model/ranking_model/sequential_1/embedding_1/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:????????? ?
Uvideo_model/ranking_model/sequential_2/integer_lookup_1/None_Lookup/LookupTableFindV2LookupTableFindV2bvideo_model_ranking_model_sequential_2_integer_lookup_1_none_lookup_lookuptablefindv2_table_handlevideo_idcvideo_model_ranking_model_sequential_2_integer_lookup_1_none_lookup_lookuptablefindv2_default_value*	
Tin0	*

Tout0	*#
_output_shapes
:??????????
@video_model/ranking_model/sequential_2/integer_lookup_1/IdentityIdentity^video_model/ranking_model/sequential_2/integer_lookup_1/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:??????????
Cvideo_model/ranking_model/sequential_2/embedding_2/embedding_lookupResourceGatherIvideo_model_ranking_model_sequential_2_embedding_2_embedding_lookup_13904Ivideo_model/ranking_model/sequential_2/integer_lookup_1/Identity:output:0*
Tindices0	*\
_classR
PNloc:@video_model/ranking_model/sequential_2/embedding_2/embedding_lookup/13904*'
_output_shapes
:????????? *
dtype0?
Lvideo_model/ranking_model/sequential_2/embedding_2/embedding_lookup/IdentityIdentityLvideo_model/ranking_model/sequential_2/embedding_2/embedding_lookup:output:0*
T0*\
_classR
PNloc:@video_model/ranking_model/sequential_2/embedding_2/embedding_lookup/13904*'
_output_shapes
:????????? ?
Nvideo_model/ranking_model/sequential_2/embedding_2/embedding_lookup/Identity_1IdentityUvideo_model/ranking_model/sequential_2/embedding_2/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:????????? g
%video_model/ranking_model/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :?
 video_model/ranking_model/concatConcatV2Svideo_model/ranking_model/sequential/embedding/embedding_lookup/Identity_1:output:0Wvideo_model/ranking_model/sequential_1/embedding_1/embedding_lookup/Identity_1:output:0.video_model/ranking_model/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????@?
Bvideo_model/ranking_model/sequential_3/dense/MatMul/ReadVariableOpReadVariableOpKvideo_model_ranking_model_sequential_3_dense_matmul_readvariableop_resource*
_output_shapes
:	@?*
dtype0?
3video_model/ranking_model/sequential_3/dense/MatMulMatMul)video_model/ranking_model/concat:output:0Jvideo_model/ranking_model/sequential_3/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
Cvideo_model/ranking_model/sequential_3/dense/BiasAdd/ReadVariableOpReadVariableOpLvideo_model_ranking_model_sequential_3_dense_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
4video_model/ranking_model/sequential_3/dense/BiasAddBiasAdd=video_model/ranking_model/sequential_3/dense/MatMul:product:0Kvideo_model/ranking_model/sequential_3/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
1video_model/ranking_model/sequential_3/dense/ReluRelu=video_model/ranking_model/sequential_3/dense/BiasAdd:output:0*
T0*(
_output_shapes
:???????????
Dvideo_model/ranking_model/sequential_3/dense_1/MatMul/ReadVariableOpReadVariableOpMvideo_model_ranking_model_sequential_3_dense_1_matmul_readvariableop_resource*
_output_shapes
:	?@*
dtype0?
5video_model/ranking_model/sequential_3/dense_1/MatMulMatMul?video_model/ranking_model/sequential_3/dense/Relu:activations:0Lvideo_model/ranking_model/sequential_3/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
Evideo_model/ranking_model/sequential_3/dense_1/BiasAdd/ReadVariableOpReadVariableOpNvideo_model_ranking_model_sequential_3_dense_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
6video_model/ranking_model/sequential_3/dense_1/BiasAddBiasAdd?video_model/ranking_model/sequential_3/dense_1/MatMul:product:0Mvideo_model/ranking_model/sequential_3/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
3video_model/ranking_model/sequential_3/dense_1/ReluRelu?video_model/ranking_model/sequential_3/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@?
Dvideo_model/ranking_model/sequential_3/dense_2/MatMul/ReadVariableOpReadVariableOpMvideo_model_ranking_model_sequential_3_dense_2_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0?
5video_model/ranking_model/sequential_3/dense_2/MatMulMatMulAvideo_model/ranking_model/sequential_3/dense_1/Relu:activations:0Lvideo_model/ranking_model/sequential_3/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
Evideo_model/ranking_model/sequential_3/dense_2/BiasAdd/ReadVariableOpReadVariableOpNvideo_model_ranking_model_sequential_3_dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
6video_model/ranking_model/sequential_3/dense_2/BiasAddBiasAdd?video_model/ranking_model/sequential_3/dense_2/MatMul:product:0Mvideo_model/ranking_model/sequential_3/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
IdentityIdentity?video_model/ranking_model/sequential_3/dense_2/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp@^video_model/ranking_model/sequential/embedding/embedding_lookupR^video_model/ranking_model/sequential/integer_lookup/None_Lookup/LookupTableFindV2D^video_model/ranking_model/sequential_1/embedding_1/embedding_lookupS^video_model/ranking_model/sequential_1/string_lookup/None_Lookup/LookupTableFindV2D^video_model/ranking_model/sequential_2/embedding_2/embedding_lookupV^video_model/ranking_model/sequential_2/integer_lookup_1/None_Lookup/LookupTableFindV2D^video_model/ranking_model/sequential_3/dense/BiasAdd/ReadVariableOpC^video_model/ranking_model/sequential_3/dense/MatMul/ReadVariableOpF^video_model/ranking_model/sequential_3/dense_1/BiasAdd/ReadVariableOpE^video_model/ranking_model/sequential_3/dense_1/MatMul/ReadVariableOpF^video_model/ranking_model/sequential_3/dense_2/BiasAdd/ReadVariableOpE^video_model/ranking_model/sequential_3/dense_2/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z:?????????:?????????:?????????:?????????: : : : : : : : : : : : : : : 2?
?video_model/ranking_model/sequential/embedding/embedding_lookup?video_model/ranking_model/sequential/embedding/embedding_lookup2?
Qvideo_model/ranking_model/sequential/integer_lookup/None_Lookup/LookupTableFindV2Qvideo_model/ranking_model/sequential/integer_lookup/None_Lookup/LookupTableFindV22?
Cvideo_model/ranking_model/sequential_1/embedding_1/embedding_lookupCvideo_model/ranking_model/sequential_1/embedding_1/embedding_lookup2?
Rvideo_model/ranking_model/sequential_1/string_lookup/None_Lookup/LookupTableFindV2Rvideo_model/ranking_model/sequential_1/string_lookup/None_Lookup/LookupTableFindV22?
Cvideo_model/ranking_model/sequential_2/embedding_2/embedding_lookupCvideo_model/ranking_model/sequential_2/embedding_2/embedding_lookup2?
Uvideo_model/ranking_model/sequential_2/integer_lookup_1/None_Lookup/LookupTableFindV2Uvideo_model/ranking_model/sequential_2/integer_lookup_1/None_Lookup/LookupTableFindV22?
Cvideo_model/ranking_model/sequential_3/dense/BiasAdd/ReadVariableOpCvideo_model/ranking_model/sequential_3/dense/BiasAdd/ReadVariableOp2?
Bvideo_model/ranking_model/sequential_3/dense/MatMul/ReadVariableOpBvideo_model/ranking_model/sequential_3/dense/MatMul/ReadVariableOp2?
Evideo_model/ranking_model/sequential_3/dense_1/BiasAdd/ReadVariableOpEvideo_model/ranking_model/sequential_3/dense_1/BiasAdd/ReadVariableOp2?
Dvideo_model/ranking_model/sequential_3/dense_1/MatMul/ReadVariableOpDvideo_model/ranking_model/sequential_3/dense_1/MatMul/ReadVariableOp2?
Evideo_model/ranking_model/sequential_3/dense_2/BiasAdd/ReadVariableOpEvideo_model/ranking_model/sequential_3/dense_2/BiasAdd/ReadVariableOp2?
Dvideo_model/ranking_model/sequential_3/dense_2/MatMul/ReadVariableOpDvideo_model/ranking_model/sequential_3/dense_2/MatMul/ReadVariableOp:L H
#
_output_shapes
:?????????
!
_user_specified_name	user_id:VR
#
_output_shapes
:?????????
+
_user_specified_namevideo_description:MI
#
_output_shapes
:?????????
"
_user_specified_name
video_id:QM
#
_output_shapes
:?????????
&
_user_specified_namevideo_rating:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
*__inference_sequential_layer_call_fn_15553

inputs	
unknown
	unknown_0	
	unknown_1:$ 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_14024o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:????????? `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: 
?
?
E__inference_sequential_layer_call_and_return_conditional_losses_13983

inputs	=
9integer_lookup_none_lookup_lookuptablefindv2_table_handle>
:integer_lookup_none_lookup_lookuptablefindv2_default_value	!
embedding_13979:$ 
identity??!embedding/StatefulPartitionedCall?,integer_lookup/None_Lookup/LookupTableFindV2?
,integer_lookup/None_Lookup/LookupTableFindV2LookupTableFindV29integer_lookup_none_lookup_lookuptablefindv2_table_handleinputs:integer_lookup_none_lookup_lookuptablefindv2_default_value*	
Tin0	*

Tout0	*#
_output_shapes
:??????????
integer_lookup/IdentityIdentity5integer_lookup/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:??????????
!embedding/StatefulPartitionedCallStatefulPartitionedCall integer_lookup/Identity:output:0embedding_13979*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_embedding_layer_call_and_return_conditional_losses_13978y
IdentityIdentity*embedding/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:????????? ?
NoOpNoOp"^embedding/StatefulPartitionedCall-^integer_lookup/None_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????: : : 2F
!embedding/StatefulPartitionedCall!embedding/StatefulPartitionedCall2\
,integer_lookup/None_Lookup/LookupTableFindV2,integer_lookup/None_Lookup/LookupTableFindV2:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: 
?
?
-__inference_ranking_model_layer_call_fn_15384
inputs_0	
inputs_1	
inputs_2
unknown
	unknown_0	
	unknown_1:$ 
	unknown_2
	unknown_3	
	unknown_4: 
	unknown_5
	unknown_6	
	unknown_7:	? 
	unknown_8:	@?
	unknown_9:	?

unknown_10:	?@

unknown_11:@

unknown_12:@

unknown_13:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1inputs_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13*
Tin
2					*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*+
_read_only_resource_inputs
	*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_ranking_model_layer_call_and_return_conditional_losses_14541o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*^
_input_shapesM
K:?????????:?????????:?????????: : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:M I
#
_output_shapes
:?????????
"
_user_specified_name
inputs/0:MI
#
_output_shapes
:?????????
"
_user_specified_name
inputs/1:MI
#
_output_shapes
:?????????
"
_user_specified_name
inputs/2:

_output_shapes
: :

_output_shapes
: :


_output_shapes
: 
?
?
,__inference_sequential_3_layer_call_fn_15692

inputs
unknown:	@?
	unknown_0:	?
	unknown_1:	?@
	unknown_2:@
	unknown_3:@
	unknown_4:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_sequential_3_layer_call_and_return_conditional_losses_14340o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????@: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
+__inference_video_model_layer_call_fn_15197
features_user_id	
features_video_description
features_video_id	
features_video_rating
unknown
	unknown_0	
	unknown_1:$ 
	unknown_2
	unknown_3	
	unknown_4: 
	unknown_5
	unknown_6	
	unknown_7:	? 
	unknown_8:	@?
	unknown_9:	?

unknown_10:	?@

unknown_11:@

unknown_12:@

unknown_13:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallfeatures_user_idfeatures_video_descriptionfeatures_video_idfeatures_video_ratingunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13*
Tin
2					*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*+
_read_only_resource_inputs
		*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_video_model_layer_call_and_return_conditional_losses_14856o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z:?????????:?????????:?????????:?????????: : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
#
_output_shapes
:?????????
*
_user_specified_namefeatures/user_id:_[
#
_output_shapes
:?????????
4
_user_specified_namefeatures/video_description:VR
#
_output_shapes
:?????????
+
_user_specified_namefeatures/video_id:ZV
#
_output_shapes
:?????????
/
_user_specified_namefeatures/video_rating:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?

?
B__inference_dense_1_layer_call_and_return_conditional_losses_14317

inputs1
matmul_readvariableop_resource:	?@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????@a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????@w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
H__inference_ranking_model_layer_call_and_return_conditional_losses_14657

inputs	
inputs_1	
inputs_2
sequential_14620
sequential_14622	"
sequential_14624:$ 
sequential_1_14627
sequential_1_14629	$
sequential_1_14631: 
sequential_2_14634
sequential_2_14636	%
sequential_2_14638:	? %
sequential_3_14643:	@?!
sequential_3_14645:	?%
sequential_3_14647:	?@ 
sequential_3_14649:@$
sequential_3_14651:@ 
sequential_3_14653:
identity??"sequential/StatefulPartitionedCall?$sequential_1/StatefulPartitionedCall?$sequential_2/StatefulPartitionedCall?$sequential_3/StatefulPartitionedCall?
"sequential/StatefulPartitionedCallStatefulPartitionedCallinputssequential_14620sequential_14622sequential_14624*
Tin
2		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_14024?
$sequential_1/StatefulPartitionedCallStatefulPartitionedCallinputs_2sequential_1_14627sequential_1_14629sequential_1_14631*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_sequential_1_layer_call_and_return_conditional_losses_14132?
$sequential_2/StatefulPartitionedCallStatefulPartitionedCallinputs_1sequential_2_14634sequential_2_14636sequential_2_14638*
Tin
2		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_sequential_2_layer_call_and_return_conditional_losses_14240M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :?
concatConcatV2+sequential/StatefulPartitionedCall:output:0-sequential_1/StatefulPartitionedCall:output:0concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????@?
$sequential_3/StatefulPartitionedCallStatefulPartitionedCallconcat:output:0sequential_3_14643sequential_3_14645sequential_3_14647sequential_3_14649sequential_3_14651sequential_3_14653*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_sequential_3_layer_call_and_return_conditional_losses_14423|
IdentityIdentity-sequential_3/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp#^sequential/StatefulPartitionedCall%^sequential_1/StatefulPartitionedCall%^sequential_2/StatefulPartitionedCall%^sequential_3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*^
_input_shapesM
K:?????????:?????????:?????????: : : : : : : : : : : : : : : 2H
"sequential/StatefulPartitionedCall"sequential/StatefulPartitionedCall2L
$sequential_1/StatefulPartitionedCall$sequential_1/StatefulPartitionedCall2L
$sequential_2/StatefulPartitionedCall$sequential_2/StatefulPartitionedCall2L
$sequential_3/StatefulPartitionedCall$sequential_3/StatefulPartitionedCall:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs:KG
#
_output_shapes
:?????????
 
_user_specified_nameinputs:KG
#
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :


_output_shapes
: 
?	
?
,__inference_sequential_3_layer_call_fn_14355
dense_input
unknown:	@?
	unknown_0:	?
	unknown_1:	?@
	unknown_2:@
	unknown_3:@
	unknown_4:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalldense_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_sequential_3_layer_call_and_return_conditional_losses_14340o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????@: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
'
_output_shapes
:?????????@
%
_user_specified_namedense_input
?
?
G__inference_sequential_1_layer_call_and_return_conditional_losses_14132

inputs<
8string_lookup_none_lookup_lookuptablefindv2_table_handle=
9string_lookup_none_lookup_lookuptablefindv2_default_value	#
embedding_1_14128: 
identity??#embedding_1/StatefulPartitionedCall?+string_lookup/None_Lookup/LookupTableFindV2?
+string_lookup/None_Lookup/LookupTableFindV2LookupTableFindV28string_lookup_none_lookup_lookuptablefindv2_table_handleinputs9string_lookup_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
string_lookup/IdentityIdentity4string_lookup/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:??????????
#embedding_1/StatefulPartitionedCallStatefulPartitionedCallstring_lookup/Identity:output:0embedding_1_14128*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_embedding_1_layer_call_and_return_conditional_losses_14086{
IdentityIdentity,embedding_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:????????? ?
NoOpNoOp$^embedding_1/StatefulPartitionedCall,^string_lookup/None_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????: : : 2J
#embedding_1/StatefulPartitionedCall#embedding_1/StatefulPartitionedCall2Z
+string_lookup/None_Lookup/LookupTableFindV2+string_lookup/None_Lookup/LookupTableFindV2:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: 
?
:
__inference__creator_15887
identity??
hash_tablej

hash_tableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name82*
value_dtype0	W
IdentityIdentityhash_table:table_handle:0^NoOp*
T0*
_output_shapes
: S
NoOpNoOp^hash_table*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 2

hash_table
hash_table
?
?
F__inference_embedding_1_layer_call_and_return_conditional_losses_15789

inputs	(
embedding_lookup_15783: 
identity??embedding_lookup?
embedding_lookupResourceGatherembedding_lookup_15783inputs*
Tindices0	*)
_class
loc:@embedding_lookup/15783*'
_output_shapes
:????????? *
dtype0?
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*)
_class
loc:@embedding_lookup/15783*'
_output_shapes
:????????? }
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:????????? s
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*'
_output_shapes
:????????? Y
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:?????????: 2$
embedding_lookupembedding_lookup:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
D__inference_embedding_layer_call_and_return_conditional_losses_13978

inputs	(
embedding_lookup_13972:$ 
identity??embedding_lookup?
embedding_lookupResourceGatherembedding_lookup_13972inputs*
Tindices0	*)
_class
loc:@embedding_lookup/13972*'
_output_shapes
:????????? *
dtype0?
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*)
_class
loc:@embedding_lookup/13972*'
_output_shapes
:????????? }
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:????????? s
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*'
_output_shapes
:????????? Y
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:?????????: 2$
embedding_lookupembedding_lookup:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
F__inference_embedding_2_layer_call_and_return_conditional_losses_14194

inputs	)
embedding_lookup_14188:	? 
identity??embedding_lookup?
embedding_lookupResourceGatherembedding_lookup_14188inputs*
Tindices0	*)
_class
loc:@embedding_lookup/14188*'
_output_shapes
:????????? *
dtype0?
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*)
_class
loc:@embedding_lookup/14188*'
_output_shapes
:????????? }
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:????????? s
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*'
_output_shapes
:????????? Y
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:?????????: 2$
embedding_lookupembedding_lookup:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
G__inference_sequential_2_layer_call_and_return_conditional_losses_15675

inputs	?
;integer_lookup_1_none_lookup_lookuptablefindv2_table_handle@
<integer_lookup_1_none_lookup_lookuptablefindv2_default_value	5
"embedding_2_embedding_lookup_15669:	? 
identity??embedding_2/embedding_lookup?.integer_lookup_1/None_Lookup/LookupTableFindV2?
.integer_lookup_1/None_Lookup/LookupTableFindV2LookupTableFindV2;integer_lookup_1_none_lookup_lookuptablefindv2_table_handleinputs<integer_lookup_1_none_lookup_lookuptablefindv2_default_value*	
Tin0	*

Tout0	*#
_output_shapes
:??????????
integer_lookup_1/IdentityIdentity7integer_lookup_1/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:??????????
embedding_2/embedding_lookupResourceGather"embedding_2_embedding_lookup_15669"integer_lookup_1/Identity:output:0*
Tindices0	*5
_class+
)'loc:@embedding_2/embedding_lookup/15669*'
_output_shapes
:????????? *
dtype0?
%embedding_2/embedding_lookup/IdentityIdentity%embedding_2/embedding_lookup:output:0*
T0*5
_class+
)'loc:@embedding_2/embedding_lookup/15669*'
_output_shapes
:????????? ?
'embedding_2/embedding_lookup/Identity_1Identity.embedding_2/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:????????? 
IdentityIdentity0embedding_2/embedding_lookup/Identity_1:output:0^NoOp*
T0*'
_output_shapes
:????????? ?
NoOpNoOp^embedding_2/embedding_lookup/^integer_lookup_1/None_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????: : : 2<
embedding_2/embedding_lookupembedding_2/embedding_lookup2`
.integer_lookup_1/None_Lookup/LookupTableFindV2.integer_lookup_1/None_Lookup/LookupTableFindV2:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: 
?
?
,__inference_sequential_1_layer_call_fn_14152
string_lookup_input
unknown
	unknown_0	
	unknown_1: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallstring_lookup_inputunknown	unknown_0	unknown_1*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_sequential_1_layer_call_and_return_conditional_losses_14132o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:????????? `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
#
_output_shapes
:?????????
-
_user_specified_namestring_lookup_input:

_output_shapes
: 
?
?
E__inference_sequential_layer_call_and_return_conditional_losses_15566

inputs	=
9integer_lookup_none_lookup_lookuptablefindv2_table_handle>
:integer_lookup_none_lookup_lookuptablefindv2_default_value	2
 embedding_embedding_lookup_15560:$ 
identity??embedding/embedding_lookup?,integer_lookup/None_Lookup/LookupTableFindV2?
,integer_lookup/None_Lookup/LookupTableFindV2LookupTableFindV29integer_lookup_none_lookup_lookuptablefindv2_table_handleinputs:integer_lookup_none_lookup_lookuptablefindv2_default_value*	
Tin0	*

Tout0	*#
_output_shapes
:??????????
integer_lookup/IdentityIdentity5integer_lookup/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:??????????
embedding/embedding_lookupResourceGather embedding_embedding_lookup_15560 integer_lookup/Identity:output:0*
Tindices0	*3
_class)
'%loc:@embedding/embedding_lookup/15560*'
_output_shapes
:????????? *
dtype0?
#embedding/embedding_lookup/IdentityIdentity#embedding/embedding_lookup:output:0*
T0*3
_class)
'%loc:@embedding/embedding_lookup/15560*'
_output_shapes
:????????? ?
%embedding/embedding_lookup/Identity_1Identity,embedding/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:????????? }
IdentityIdentity.embedding/embedding_lookup/Identity_1:output:0^NoOp*
T0*'
_output_shapes
:????????? ?
NoOpNoOp^embedding/embedding_lookup-^integer_lookup/None_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????: : : 28
embedding/embedding_lookupembedding/embedding_lookup2\
,integer_lookup/None_Lookup/LookupTableFindV2,integer_lookup/None_Lookup/LookupTableFindV2:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: 
?
:
__inference__creator_15869
identity??
hash_tablej

hash_tableHashTableV2*
_output_shapes
: *
	key_dtype0	*
shared_name60*
value_dtype0	W
IdentityIdentityhash_table:table_handle:0^NoOp*
T0*
_output_shapes
: S
NoOpNoOp^hash_table*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 2

hash_table
hash_table
?
?
,__inference_sequential_2_layer_call_fn_15638

inputs	
unknown
	unknown_0	
	unknown_1:	? 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_sequential_2_layer_call_and_return_conditional_losses_14199o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:????????? `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: 
?
?
G__inference_sequential_2_layer_call_and_return_conditional_losses_14240

inputs	?
;integer_lookup_1_none_lookup_lookuptablefindv2_table_handle@
<integer_lookup_1_none_lookup_lookuptablefindv2_default_value	$
embedding_2_14236:	? 
identity??#embedding_2/StatefulPartitionedCall?.integer_lookup_1/None_Lookup/LookupTableFindV2?
.integer_lookup_1/None_Lookup/LookupTableFindV2LookupTableFindV2;integer_lookup_1_none_lookup_lookuptablefindv2_table_handleinputs<integer_lookup_1_none_lookup_lookuptablefindv2_default_value*	
Tin0	*

Tout0	*#
_output_shapes
:??????????
integer_lookup_1/IdentityIdentity7integer_lookup_1/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:??????????
#embedding_2/StatefulPartitionedCallStatefulPartitionedCall"integer_lookup_1/Identity:output:0embedding_2_14236*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_embedding_2_layer_call_and_return_conditional_losses_14194{
IdentityIdentity,embedding_2/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:????????? ?
NoOpNoOp$^embedding_2/StatefulPartitionedCall/^integer_lookup_1/None_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????: : : 2J
#embedding_2/StatefulPartitionedCall#embedding_2/StatefulPartitionedCall2`
.integer_lookup_1/None_Lookup/LookupTableFindV2.integer_lookup_1/None_Lookup/LookupTableFindV2:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: 
?
?
,__inference_sequential_1_layer_call_fn_15590

inputs
unknown
	unknown_0	
	unknown_1: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_sequential_1_layer_call_and_return_conditional_losses_14091o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:????????? `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: 
?a
?
F__inference_video_model_layer_call_and_return_conditional_losses_15347
features_user_id	
features_video_description
features_video_id	
features_video_ratingV
Rranking_model_sequential_integer_lookup_none_lookup_lookuptablefindv2_table_handleW
Sranking_model_sequential_integer_lookup_none_lookup_lookuptablefindv2_default_value	K
9ranking_model_sequential_embedding_embedding_lookup_15301:$ W
Sranking_model_sequential_1_string_lookup_none_lookup_lookuptablefindv2_table_handleX
Tranking_model_sequential_1_string_lookup_none_lookup_lookuptablefindv2_default_value	O
=ranking_model_sequential_1_embedding_1_embedding_lookup_15310: Z
Vranking_model_sequential_2_integer_lookup_1_none_lookup_lookuptablefindv2_table_handle[
Wranking_model_sequential_2_integer_lookup_1_none_lookup_lookuptablefindv2_default_value	P
=ranking_model_sequential_2_embedding_2_embedding_lookup_15319:	? R
?ranking_model_sequential_3_dense_matmul_readvariableop_resource:	@?O
@ranking_model_sequential_3_dense_biasadd_readvariableop_resource:	?T
Aranking_model_sequential_3_dense_1_matmul_readvariableop_resource:	?@P
Branking_model_sequential_3_dense_1_biasadd_readvariableop_resource:@S
Aranking_model_sequential_3_dense_2_matmul_readvariableop_resource:@P
Branking_model_sequential_3_dense_2_biasadd_readvariableop_resource:
identity??3ranking_model/sequential/embedding/embedding_lookup?Eranking_model/sequential/integer_lookup/None_Lookup/LookupTableFindV2?7ranking_model/sequential_1/embedding_1/embedding_lookup?Franking_model/sequential_1/string_lookup/None_Lookup/LookupTableFindV2?7ranking_model/sequential_2/embedding_2/embedding_lookup?Iranking_model/sequential_2/integer_lookup_1/None_Lookup/LookupTableFindV2?7ranking_model/sequential_3/dense/BiasAdd/ReadVariableOp?6ranking_model/sequential_3/dense/MatMul/ReadVariableOp?9ranking_model/sequential_3/dense_1/BiasAdd/ReadVariableOp?8ranking_model/sequential_3/dense_1/MatMul/ReadVariableOp?9ranking_model/sequential_3/dense_2/BiasAdd/ReadVariableOp?8ranking_model/sequential_3/dense_2/MatMul/ReadVariableOp?
Eranking_model/sequential/integer_lookup/None_Lookup/LookupTableFindV2LookupTableFindV2Rranking_model_sequential_integer_lookup_none_lookup_lookuptablefindv2_table_handlefeatures_user_idSranking_model_sequential_integer_lookup_none_lookup_lookuptablefindv2_default_value*	
Tin0	*

Tout0	*#
_output_shapes
:??????????
0ranking_model/sequential/integer_lookup/IdentityIdentityNranking_model/sequential/integer_lookup/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:??????????
3ranking_model/sequential/embedding/embedding_lookupResourceGather9ranking_model_sequential_embedding_embedding_lookup_153019ranking_model/sequential/integer_lookup/Identity:output:0*
Tindices0	*L
_classB
@>loc:@ranking_model/sequential/embedding/embedding_lookup/15301*'
_output_shapes
:????????? *
dtype0?
<ranking_model/sequential/embedding/embedding_lookup/IdentityIdentity<ranking_model/sequential/embedding/embedding_lookup:output:0*
T0*L
_classB
@>loc:@ranking_model/sequential/embedding/embedding_lookup/15301*'
_output_shapes
:????????? ?
>ranking_model/sequential/embedding/embedding_lookup/Identity_1IdentityEranking_model/sequential/embedding/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:????????? ?
Franking_model/sequential_1/string_lookup/None_Lookup/LookupTableFindV2LookupTableFindV2Sranking_model_sequential_1_string_lookup_none_lookup_lookuptablefindv2_table_handlefeatures_video_descriptionTranking_model_sequential_1_string_lookup_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
1ranking_model/sequential_1/string_lookup/IdentityIdentityOranking_model/sequential_1/string_lookup/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:??????????
7ranking_model/sequential_1/embedding_1/embedding_lookupResourceGather=ranking_model_sequential_1_embedding_1_embedding_lookup_15310:ranking_model/sequential_1/string_lookup/Identity:output:0*
Tindices0	*P
_classF
DBloc:@ranking_model/sequential_1/embedding_1/embedding_lookup/15310*'
_output_shapes
:????????? *
dtype0?
@ranking_model/sequential_1/embedding_1/embedding_lookup/IdentityIdentity@ranking_model/sequential_1/embedding_1/embedding_lookup:output:0*
T0*P
_classF
DBloc:@ranking_model/sequential_1/embedding_1/embedding_lookup/15310*'
_output_shapes
:????????? ?
Branking_model/sequential_1/embedding_1/embedding_lookup/Identity_1IdentityIranking_model/sequential_1/embedding_1/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:????????? ?
Iranking_model/sequential_2/integer_lookup_1/None_Lookup/LookupTableFindV2LookupTableFindV2Vranking_model_sequential_2_integer_lookup_1_none_lookup_lookuptablefindv2_table_handlefeatures_video_idWranking_model_sequential_2_integer_lookup_1_none_lookup_lookuptablefindv2_default_value*	
Tin0	*

Tout0	*#
_output_shapes
:??????????
4ranking_model/sequential_2/integer_lookup_1/IdentityIdentityRranking_model/sequential_2/integer_lookup_1/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:??????????
7ranking_model/sequential_2/embedding_2/embedding_lookupResourceGather=ranking_model_sequential_2_embedding_2_embedding_lookup_15319=ranking_model/sequential_2/integer_lookup_1/Identity:output:0*
Tindices0	*P
_classF
DBloc:@ranking_model/sequential_2/embedding_2/embedding_lookup/15319*'
_output_shapes
:????????? *
dtype0?
@ranking_model/sequential_2/embedding_2/embedding_lookup/IdentityIdentity@ranking_model/sequential_2/embedding_2/embedding_lookup:output:0*
T0*P
_classF
DBloc:@ranking_model/sequential_2/embedding_2/embedding_lookup/15319*'
_output_shapes
:????????? ?
Branking_model/sequential_2/embedding_2/embedding_lookup/Identity_1IdentityIranking_model/sequential_2/embedding_2/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:????????? [
ranking_model/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :?
ranking_model/concatConcatV2Granking_model/sequential/embedding/embedding_lookup/Identity_1:output:0Kranking_model/sequential_1/embedding_1/embedding_lookup/Identity_1:output:0"ranking_model/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????@?
6ranking_model/sequential_3/dense/MatMul/ReadVariableOpReadVariableOp?ranking_model_sequential_3_dense_matmul_readvariableop_resource*
_output_shapes
:	@?*
dtype0?
'ranking_model/sequential_3/dense/MatMulMatMulranking_model/concat:output:0>ranking_model/sequential_3/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
7ranking_model/sequential_3/dense/BiasAdd/ReadVariableOpReadVariableOp@ranking_model_sequential_3_dense_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
(ranking_model/sequential_3/dense/BiasAddBiasAdd1ranking_model/sequential_3/dense/MatMul:product:0?ranking_model/sequential_3/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
%ranking_model/sequential_3/dense/ReluRelu1ranking_model/sequential_3/dense/BiasAdd:output:0*
T0*(
_output_shapes
:???????????
8ranking_model/sequential_3/dense_1/MatMul/ReadVariableOpReadVariableOpAranking_model_sequential_3_dense_1_matmul_readvariableop_resource*
_output_shapes
:	?@*
dtype0?
)ranking_model/sequential_3/dense_1/MatMulMatMul3ranking_model/sequential_3/dense/Relu:activations:0@ranking_model/sequential_3/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
9ranking_model/sequential_3/dense_1/BiasAdd/ReadVariableOpReadVariableOpBranking_model_sequential_3_dense_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
*ranking_model/sequential_3/dense_1/BiasAddBiasAdd3ranking_model/sequential_3/dense_1/MatMul:product:0Aranking_model/sequential_3/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
'ranking_model/sequential_3/dense_1/ReluRelu3ranking_model/sequential_3/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@?
8ranking_model/sequential_3/dense_2/MatMul/ReadVariableOpReadVariableOpAranking_model_sequential_3_dense_2_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0?
)ranking_model/sequential_3/dense_2/MatMulMatMul5ranking_model/sequential_3/dense_1/Relu:activations:0@ranking_model/sequential_3/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
9ranking_model/sequential_3/dense_2/BiasAdd/ReadVariableOpReadVariableOpBranking_model_sequential_3_dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
*ranking_model/sequential_3/dense_2/BiasAddBiasAdd3ranking_model/sequential_3/dense_2/MatMul:product:0Aranking_model/sequential_3/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
IdentityIdentity3ranking_model/sequential_3/dense_2/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp4^ranking_model/sequential/embedding/embedding_lookupF^ranking_model/sequential/integer_lookup/None_Lookup/LookupTableFindV28^ranking_model/sequential_1/embedding_1/embedding_lookupG^ranking_model/sequential_1/string_lookup/None_Lookup/LookupTableFindV28^ranking_model/sequential_2/embedding_2/embedding_lookupJ^ranking_model/sequential_2/integer_lookup_1/None_Lookup/LookupTableFindV28^ranking_model/sequential_3/dense/BiasAdd/ReadVariableOp7^ranking_model/sequential_3/dense/MatMul/ReadVariableOp:^ranking_model/sequential_3/dense_1/BiasAdd/ReadVariableOp9^ranking_model/sequential_3/dense_1/MatMul/ReadVariableOp:^ranking_model/sequential_3/dense_2/BiasAdd/ReadVariableOp9^ranking_model/sequential_3/dense_2/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z:?????????:?????????:?????????:?????????: : : : : : : : : : : : : : : 2j
3ranking_model/sequential/embedding/embedding_lookup3ranking_model/sequential/embedding/embedding_lookup2?
Eranking_model/sequential/integer_lookup/None_Lookup/LookupTableFindV2Eranking_model/sequential/integer_lookup/None_Lookup/LookupTableFindV22r
7ranking_model/sequential_1/embedding_1/embedding_lookup7ranking_model/sequential_1/embedding_1/embedding_lookup2?
Franking_model/sequential_1/string_lookup/None_Lookup/LookupTableFindV2Franking_model/sequential_1/string_lookup/None_Lookup/LookupTableFindV22r
7ranking_model/sequential_2/embedding_2/embedding_lookup7ranking_model/sequential_2/embedding_2/embedding_lookup2?
Iranking_model/sequential_2/integer_lookup_1/None_Lookup/LookupTableFindV2Iranking_model/sequential_2/integer_lookup_1/None_Lookup/LookupTableFindV22r
7ranking_model/sequential_3/dense/BiasAdd/ReadVariableOp7ranking_model/sequential_3/dense/BiasAdd/ReadVariableOp2p
6ranking_model/sequential_3/dense/MatMul/ReadVariableOp6ranking_model/sequential_3/dense/MatMul/ReadVariableOp2v
9ranking_model/sequential_3/dense_1/BiasAdd/ReadVariableOp9ranking_model/sequential_3/dense_1/BiasAdd/ReadVariableOp2t
8ranking_model/sequential_3/dense_1/MatMul/ReadVariableOp8ranking_model/sequential_3/dense_1/MatMul/ReadVariableOp2v
9ranking_model/sequential_3/dense_2/BiasAdd/ReadVariableOp9ranking_model/sequential_3/dense_2/BiasAdd/ReadVariableOp2t
8ranking_model/sequential_3/dense_2/MatMul/ReadVariableOp8ranking_model/sequential_3/dense_2/MatMul/ReadVariableOp:U Q
#
_output_shapes
:?????????
*
_user_specified_namefeatures/user_id:_[
#
_output_shapes
:?????????
4
_user_specified_namefeatures/video_description:VR
#
_output_shapes
:?????????
+
_user_specified_namefeatures/video_id:ZV
#
_output_shapes
:?????????
/
_user_specified_namefeatures/video_rating:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?P
?
H__inference_ranking_model_layer_call_and_return_conditional_losses_15531
inputs_0	
inputs_1	
inputs_2H
Dsequential_integer_lookup_none_lookup_lookuptablefindv2_table_handleI
Esequential_integer_lookup_none_lookup_lookuptablefindv2_default_value	=
+sequential_embedding_embedding_lookup_15485:$ I
Esequential_1_string_lookup_none_lookup_lookuptablefindv2_table_handleJ
Fsequential_1_string_lookup_none_lookup_lookuptablefindv2_default_value	A
/sequential_1_embedding_1_embedding_lookup_15494: L
Hsequential_2_integer_lookup_1_none_lookup_lookuptablefindv2_table_handleM
Isequential_2_integer_lookup_1_none_lookup_lookuptablefindv2_default_value	B
/sequential_2_embedding_2_embedding_lookup_15503:	? D
1sequential_3_dense_matmul_readvariableop_resource:	@?A
2sequential_3_dense_biasadd_readvariableop_resource:	?F
3sequential_3_dense_1_matmul_readvariableop_resource:	?@B
4sequential_3_dense_1_biasadd_readvariableop_resource:@E
3sequential_3_dense_2_matmul_readvariableop_resource:@B
4sequential_3_dense_2_biasadd_readvariableop_resource:
identity??%sequential/embedding/embedding_lookup?7sequential/integer_lookup/None_Lookup/LookupTableFindV2?)sequential_1/embedding_1/embedding_lookup?8sequential_1/string_lookup/None_Lookup/LookupTableFindV2?)sequential_2/embedding_2/embedding_lookup?;sequential_2/integer_lookup_1/None_Lookup/LookupTableFindV2?)sequential_3/dense/BiasAdd/ReadVariableOp?(sequential_3/dense/MatMul/ReadVariableOp?+sequential_3/dense_1/BiasAdd/ReadVariableOp?*sequential_3/dense_1/MatMul/ReadVariableOp?+sequential_3/dense_2/BiasAdd/ReadVariableOp?*sequential_3/dense_2/MatMul/ReadVariableOp?
7sequential/integer_lookup/None_Lookup/LookupTableFindV2LookupTableFindV2Dsequential_integer_lookup_none_lookup_lookuptablefindv2_table_handleinputs_0Esequential_integer_lookup_none_lookup_lookuptablefindv2_default_value*	
Tin0	*

Tout0	*#
_output_shapes
:??????????
"sequential/integer_lookup/IdentityIdentity@sequential/integer_lookup/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:??????????
%sequential/embedding/embedding_lookupResourceGather+sequential_embedding_embedding_lookup_15485+sequential/integer_lookup/Identity:output:0*
Tindices0	*>
_class4
20loc:@sequential/embedding/embedding_lookup/15485*'
_output_shapes
:????????? *
dtype0?
.sequential/embedding/embedding_lookup/IdentityIdentity.sequential/embedding/embedding_lookup:output:0*
T0*>
_class4
20loc:@sequential/embedding/embedding_lookup/15485*'
_output_shapes
:????????? ?
0sequential/embedding/embedding_lookup/Identity_1Identity7sequential/embedding/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:????????? ?
8sequential_1/string_lookup/None_Lookup/LookupTableFindV2LookupTableFindV2Esequential_1_string_lookup_none_lookup_lookuptablefindv2_table_handleinputs_2Fsequential_1_string_lookup_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
#sequential_1/string_lookup/IdentityIdentityAsequential_1/string_lookup/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:??????????
)sequential_1/embedding_1/embedding_lookupResourceGather/sequential_1_embedding_1_embedding_lookup_15494,sequential_1/string_lookup/Identity:output:0*
Tindices0	*B
_class8
64loc:@sequential_1/embedding_1/embedding_lookup/15494*'
_output_shapes
:????????? *
dtype0?
2sequential_1/embedding_1/embedding_lookup/IdentityIdentity2sequential_1/embedding_1/embedding_lookup:output:0*
T0*B
_class8
64loc:@sequential_1/embedding_1/embedding_lookup/15494*'
_output_shapes
:????????? ?
4sequential_1/embedding_1/embedding_lookup/Identity_1Identity;sequential_1/embedding_1/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:????????? ?
;sequential_2/integer_lookup_1/None_Lookup/LookupTableFindV2LookupTableFindV2Hsequential_2_integer_lookup_1_none_lookup_lookuptablefindv2_table_handleinputs_1Isequential_2_integer_lookup_1_none_lookup_lookuptablefindv2_default_value*	
Tin0	*

Tout0	*#
_output_shapes
:??????????
&sequential_2/integer_lookup_1/IdentityIdentityDsequential_2/integer_lookup_1/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:??????????
)sequential_2/embedding_2/embedding_lookupResourceGather/sequential_2_embedding_2_embedding_lookup_15503/sequential_2/integer_lookup_1/Identity:output:0*
Tindices0	*B
_class8
64loc:@sequential_2/embedding_2/embedding_lookup/15503*'
_output_shapes
:????????? *
dtype0?
2sequential_2/embedding_2/embedding_lookup/IdentityIdentity2sequential_2/embedding_2/embedding_lookup:output:0*
T0*B
_class8
64loc:@sequential_2/embedding_2/embedding_lookup/15503*'
_output_shapes
:????????? ?
4sequential_2/embedding_2/embedding_lookup/Identity_1Identity;sequential_2/embedding_2/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:????????? M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :?
concatConcatV29sequential/embedding/embedding_lookup/Identity_1:output:0=sequential_1/embedding_1/embedding_lookup/Identity_1:output:0concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????@?
(sequential_3/dense/MatMul/ReadVariableOpReadVariableOp1sequential_3_dense_matmul_readvariableop_resource*
_output_shapes
:	@?*
dtype0?
sequential_3/dense/MatMulMatMulconcat:output:00sequential_3/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
)sequential_3/dense/BiasAdd/ReadVariableOpReadVariableOp2sequential_3_dense_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
sequential_3/dense/BiasAddBiasAdd#sequential_3/dense/MatMul:product:01sequential_3/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????w
sequential_3/dense/ReluRelu#sequential_3/dense/BiasAdd:output:0*
T0*(
_output_shapes
:???????????
*sequential_3/dense_1/MatMul/ReadVariableOpReadVariableOp3sequential_3_dense_1_matmul_readvariableop_resource*
_output_shapes
:	?@*
dtype0?
sequential_3/dense_1/MatMulMatMul%sequential_3/dense/Relu:activations:02sequential_3/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
+sequential_3/dense_1/BiasAdd/ReadVariableOpReadVariableOp4sequential_3_dense_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
sequential_3/dense_1/BiasAddBiasAdd%sequential_3/dense_1/MatMul:product:03sequential_3/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@z
sequential_3/dense_1/ReluRelu%sequential_3/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@?
*sequential_3/dense_2/MatMul/ReadVariableOpReadVariableOp3sequential_3_dense_2_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0?
sequential_3/dense_2/MatMulMatMul'sequential_3/dense_1/Relu:activations:02sequential_3/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
+sequential_3/dense_2/BiasAdd/ReadVariableOpReadVariableOp4sequential_3_dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
sequential_3/dense_2/BiasAddBiasAdd%sequential_3/dense_2/MatMul:product:03sequential_3/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????t
IdentityIdentity%sequential_3/dense_2/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp&^sequential/embedding/embedding_lookup8^sequential/integer_lookup/None_Lookup/LookupTableFindV2*^sequential_1/embedding_1/embedding_lookup9^sequential_1/string_lookup/None_Lookup/LookupTableFindV2*^sequential_2/embedding_2/embedding_lookup<^sequential_2/integer_lookup_1/None_Lookup/LookupTableFindV2*^sequential_3/dense/BiasAdd/ReadVariableOp)^sequential_3/dense/MatMul/ReadVariableOp,^sequential_3/dense_1/BiasAdd/ReadVariableOp+^sequential_3/dense_1/MatMul/ReadVariableOp,^sequential_3/dense_2/BiasAdd/ReadVariableOp+^sequential_3/dense_2/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*^
_input_shapesM
K:?????????:?????????:?????????: : : : : : : : : : : : : : : 2N
%sequential/embedding/embedding_lookup%sequential/embedding/embedding_lookup2r
7sequential/integer_lookup/None_Lookup/LookupTableFindV27sequential/integer_lookup/None_Lookup/LookupTableFindV22V
)sequential_1/embedding_1/embedding_lookup)sequential_1/embedding_1/embedding_lookup2t
8sequential_1/string_lookup/None_Lookup/LookupTableFindV28sequential_1/string_lookup/None_Lookup/LookupTableFindV22V
)sequential_2/embedding_2/embedding_lookup)sequential_2/embedding_2/embedding_lookup2z
;sequential_2/integer_lookup_1/None_Lookup/LookupTableFindV2;sequential_2/integer_lookup_1/None_Lookup/LookupTableFindV22V
)sequential_3/dense/BiasAdd/ReadVariableOp)sequential_3/dense/BiasAdd/ReadVariableOp2T
(sequential_3/dense/MatMul/ReadVariableOp(sequential_3/dense/MatMul/ReadVariableOp2Z
+sequential_3/dense_1/BiasAdd/ReadVariableOp+sequential_3/dense_1/BiasAdd/ReadVariableOp2X
*sequential_3/dense_1/MatMul/ReadVariableOp*sequential_3/dense_1/MatMul/ReadVariableOp2Z
+sequential_3/dense_2/BiasAdd/ReadVariableOp+sequential_3/dense_2/BiasAdd/ReadVariableOp2X
*sequential_3/dense_2/MatMul/ReadVariableOp*sequential_3/dense_2/MatMul/ReadVariableOp:M I
#
_output_shapes
:?????????
"
_user_specified_name
inputs/0:MI
#
_output_shapes
:?????????
"
_user_specified_name
inputs/1:MI
#
_output_shapes
:?????????
"
_user_specified_name
inputs/2:

_output_shapes
: :

_output_shapes
: :


_output_shapes
: 
?
?
-__inference_ranking_model_layer_call_fn_15421
inputs_0	
inputs_1	
inputs_2
unknown
	unknown_0	
	unknown_1:$ 
	unknown_2
	unknown_3	
	unknown_4: 
	unknown_5
	unknown_6	
	unknown_7:	? 
	unknown_8:	@?
	unknown_9:	?

unknown_10:	?@

unknown_11:@

unknown_12:@

unknown_13:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1inputs_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13*
Tin
2					*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*+
_read_only_resource_inputs
	*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_ranking_model_layer_call_and_return_conditional_losses_14657o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*^
_input_shapesM
K:?????????:?????????:?????????: : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:M I
#
_output_shapes
:?????????
"
_user_specified_name
inputs/0:MI
#
_output_shapes
:?????????
"
_user_specified_name
inputs/1:MI
#
_output_shapes
:?????????
"
_user_specified_name
inputs/2:

_output_shapes
: :

_output_shapes
: :


_output_shapes
: 
?
?
E__inference_sequential_layer_call_and_return_conditional_losses_14024

inputs	=
9integer_lookup_none_lookup_lookuptablefindv2_table_handle>
:integer_lookup_none_lookup_lookuptablefindv2_default_value	!
embedding_14020:$ 
identity??!embedding/StatefulPartitionedCall?,integer_lookup/None_Lookup/LookupTableFindV2?
,integer_lookup/None_Lookup/LookupTableFindV2LookupTableFindV29integer_lookup_none_lookup_lookuptablefindv2_table_handleinputs:integer_lookup_none_lookup_lookuptablefindv2_default_value*	
Tin0	*

Tout0	*#
_output_shapes
:??????????
integer_lookup/IdentityIdentity5integer_lookup/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:??????????
!embedding/StatefulPartitionedCallStatefulPartitionedCall integer_lookup/Identity:output:0embedding_14020*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_embedding_layer_call_and_return_conditional_losses_13978y
IdentityIdentity*embedding/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:????????? ?
NoOpNoOp"^embedding/StatefulPartitionedCall-^integer_lookup/None_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????: : : 2F
!embedding/StatefulPartitionedCall!embedding/StatefulPartitionedCall2\
,integer_lookup/None_Lookup/LookupTableFindV2,integer_lookup/None_Lookup/LookupTableFindV2:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: 
?
?
G__inference_sequential_2_layer_call_and_return_conditional_losses_15662

inputs	?
;integer_lookup_1_none_lookup_lookuptablefindv2_table_handle@
<integer_lookup_1_none_lookup_lookuptablefindv2_default_value	5
"embedding_2_embedding_lookup_15656:	? 
identity??embedding_2/embedding_lookup?.integer_lookup_1/None_Lookup/LookupTableFindV2?
.integer_lookup_1/None_Lookup/LookupTableFindV2LookupTableFindV2;integer_lookup_1_none_lookup_lookuptablefindv2_table_handleinputs<integer_lookup_1_none_lookup_lookuptablefindv2_default_value*	
Tin0	*

Tout0	*#
_output_shapes
:??????????
integer_lookup_1/IdentityIdentity7integer_lookup_1/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:??????????
embedding_2/embedding_lookupResourceGather"embedding_2_embedding_lookup_15656"integer_lookup_1/Identity:output:0*
Tindices0	*5
_class+
)'loc:@embedding_2/embedding_lookup/15656*'
_output_shapes
:????????? *
dtype0?
%embedding_2/embedding_lookup/IdentityIdentity%embedding_2/embedding_lookup:output:0*
T0*5
_class+
)'loc:@embedding_2/embedding_lookup/15656*'
_output_shapes
:????????? ?
'embedding_2/embedding_lookup/Identity_1Identity.embedding_2/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:????????? 
IdentityIdentity0embedding_2/embedding_lookup/Identity_1:output:0^NoOp*
T0*'
_output_shapes
:????????? ?
NoOpNoOp^embedding_2/embedding_lookup/^integer_lookup_1/None_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????: : : 2<
embedding_2/embedding_lookupembedding_2/embedding_lookup2`
.integer_lookup_1/None_Lookup/LookupTableFindV2.integer_lookup_1/None_Lookup/LookupTableFindV2:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: 
?
?
H__inference_ranking_model_layer_call_and_return_conditional_losses_14541

inputs	
inputs_1	
inputs_2
sequential_14504
sequential_14506	"
sequential_14508:$ 
sequential_1_14511
sequential_1_14513	$
sequential_1_14515: 
sequential_2_14518
sequential_2_14520	%
sequential_2_14522:	? %
sequential_3_14527:	@?!
sequential_3_14529:	?%
sequential_3_14531:	?@ 
sequential_3_14533:@$
sequential_3_14535:@ 
sequential_3_14537:
identity??"sequential/StatefulPartitionedCall?$sequential_1/StatefulPartitionedCall?$sequential_2/StatefulPartitionedCall?$sequential_3/StatefulPartitionedCall?
"sequential/StatefulPartitionedCallStatefulPartitionedCallinputssequential_14504sequential_14506sequential_14508*
Tin
2		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_13983?
$sequential_1/StatefulPartitionedCallStatefulPartitionedCallinputs_2sequential_1_14511sequential_1_14513sequential_1_14515*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_sequential_1_layer_call_and_return_conditional_losses_14091?
$sequential_2/StatefulPartitionedCallStatefulPartitionedCallinputs_1sequential_2_14518sequential_2_14520sequential_2_14522*
Tin
2		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_sequential_2_layer_call_and_return_conditional_losses_14199M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :?
concatConcatV2+sequential/StatefulPartitionedCall:output:0-sequential_1/StatefulPartitionedCall:output:0concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????@?
$sequential_3/StatefulPartitionedCallStatefulPartitionedCallconcat:output:0sequential_3_14527sequential_3_14529sequential_3_14531sequential_3_14533sequential_3_14535sequential_3_14537*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_sequential_3_layer_call_and_return_conditional_losses_14340|
IdentityIdentity-sequential_3/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp#^sequential/StatefulPartitionedCall%^sequential_1/StatefulPartitionedCall%^sequential_2/StatefulPartitionedCall%^sequential_3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*^
_input_shapesM
K:?????????:?????????:?????????: : : : : : : : : : : : : : : 2H
"sequential/StatefulPartitionedCall"sequential/StatefulPartitionedCall2L
$sequential_1/StatefulPartitionedCall$sequential_1/StatefulPartitionedCall2L
$sequential_2/StatefulPartitionedCall$sequential_2/StatefulPartitionedCall2L
$sequential_3/StatefulPartitionedCall$sequential_3/StatefulPartitionedCall:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs:KG
#
_output_shapes
:?????????
 
_user_specified_nameinputs:KG
#
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :


_output_shapes
: 
?
?
*__inference_sequential_layer_call_fn_13992
integer_lookup_input	
unknown
	unknown_0	
	unknown_1:$ 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinteger_lookup_inputunknown	unknown_0	unknown_1*
Tin
2		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_13983o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:????????? `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
#
_output_shapes
:?????????
.
_user_specified_nameinteger_lookup_input:

_output_shapes
: 
?
?
G__inference_sequential_3_layer_call_and_return_conditional_losses_14493
dense_input
dense_14477:	@?
dense_14479:	? 
dense_1_14482:	?@
dense_1_14484:@
dense_2_14487:@
dense_2_14489:
identity??dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?dense_2/StatefulPartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCalldense_inputdense_14477dense_14479*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_14300?
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_14482dense_1_14484*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_14317?
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_14487dense_2_14489*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_14333w
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????@: : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall:T P
'
_output_shapes
:?????????@
%
_user_specified_namedense_input
?
?
__inference__initializer_158955
1key_value_init81_lookuptableimportv2_table_handle-
)key_value_init81_lookuptableimportv2_keys/
+key_value_init81_lookuptableimportv2_values	
identity??$key_value_init81/LookupTableImportV2?
$key_value_init81/LookupTableImportV2LookupTableImportV21key_value_init81_lookuptableimportv2_table_handle)key_value_init81_lookuptableimportv2_keys+key_value_init81_lookuptableimportv2_values*	
Tin0*

Tout0	*
_output_shapes
 G
ConstConst*
_output_shapes
: *
dtype0*
value	B :L
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: m
NoOpNoOp%^key_value_init81/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
: :~:~2L
$key_value_init81/LookupTableImportV2$key_value_init81/LookupTableImportV2: 

_output_shapes
:~: 

_output_shapes
:~
?
?
'__inference_dense_2_layer_call_fn_15854

inputs
unknown:@
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_14333o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????@: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
}
)__inference_embedding_layer_call_fn_15764

inputs	
unknown:$ 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_embedding_layer_call_and_return_conditional_losses_13978o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:????????? `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:?????????: 22
StatefulPartitionedCallStatefulPartitionedCall:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
*__inference_sequential_layer_call_fn_15542

inputs	
unknown
	unknown_0	
	unknown_1:$ 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_13983o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:????????? `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: 
?	
?
B__inference_dense_2_layer_call_and_return_conditional_losses_14333

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
F__inference_video_model_layer_call_and_return_conditional_losses_14856
features	

features_1

features_2	

features_3
ranking_model_14824
ranking_model_14826	%
ranking_model_14828:$ 
ranking_model_14830
ranking_model_14832	%
ranking_model_14834: 
ranking_model_14836
ranking_model_14838	&
ranking_model_14840:	? &
ranking_model_14842:	@?"
ranking_model_14844:	?&
ranking_model_14846:	?@!
ranking_model_14848:@%
ranking_model_14850:@!
ranking_model_14852:
identity??%ranking_model/StatefulPartitionedCall?
%ranking_model/StatefulPartitionedCallStatefulPartitionedCallfeatures
features_2
features_1ranking_model_14824ranking_model_14826ranking_model_14828ranking_model_14830ranking_model_14832ranking_model_14834ranking_model_14836ranking_model_14838ranking_model_14840ranking_model_14842ranking_model_14844ranking_model_14846ranking_model_14848ranking_model_14850ranking_model_14852*
Tin
2					*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*+
_read_only_resource_inputs
	*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_ranking_model_layer_call_and_return_conditional_losses_14541}
IdentityIdentity.ranking_model/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????n
NoOpNoOp&^ranking_model/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z:?????????:?????????:?????????:?????????: : : : : : : : : : : : : : : 2N
%ranking_model/StatefulPartitionedCall%ranking_model/StatefulPartitionedCall:M I
#
_output_shapes
:?????????
"
_user_specified_name
features:MI
#
_output_shapes
:?????????
"
_user_specified_name
features:MI
#
_output_shapes
:?????????
"
_user_specified_name
features:MI
#
_output_shapes
:?????????
"
_user_specified_name
features:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
__inference__initializer_159136
2key_value_init103_lookuptableimportv2_table_handle.
*key_value_init103_lookuptableimportv2_keys	0
,key_value_init103_lookuptableimportv2_values	
identity??%key_value_init103/LookupTableImportV2?
%key_value_init103/LookupTableImportV2LookupTableImportV22key_value_init103_lookuptableimportv2_table_handle*key_value_init103_lookuptableimportv2_keys,key_value_init103_lookuptableimportv2_values*	
Tin0	*

Tout0	*
_output_shapes
 G
ConstConst*
_output_shapes
: *
dtype0*
value	B :L
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: n
NoOpNoOp&^key_value_init103/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*#
_input_shapes
: :?:?2N
%key_value_init103/LookupTableImportV2%key_value_init103/LookupTableImportV2:!

_output_shapes	
:?:!

_output_shapes	
:?
?
?
'__inference_dense_1_layer_call_fn_15834

inputs
unknown:	?@
	unknown_0:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_14317o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
F__inference_video_model_layer_call_and_return_conditional_losses_15117
user_id	
video_description
video_id	
video_rating
ranking_model_15085
ranking_model_15087	%
ranking_model_15089:$ 
ranking_model_15091
ranking_model_15093	%
ranking_model_15095: 
ranking_model_15097
ranking_model_15099	&
ranking_model_15101:	? &
ranking_model_15103:	@?"
ranking_model_15105:	?&
ranking_model_15107:	?@!
ranking_model_15109:@%
ranking_model_15111:@!
ranking_model_15113:
identity??%ranking_model/StatefulPartitionedCall?
%ranking_model/StatefulPartitionedCallStatefulPartitionedCalluser_idvideo_idvideo_descriptionranking_model_15085ranking_model_15087ranking_model_15089ranking_model_15091ranking_model_15093ranking_model_15095ranking_model_15097ranking_model_15099ranking_model_15101ranking_model_15103ranking_model_15105ranking_model_15107ranking_model_15109ranking_model_15111ranking_model_15113*
Tin
2					*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*+
_read_only_resource_inputs
	*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_ranking_model_layer_call_and_return_conditional_losses_14657}
IdentityIdentity.ranking_model/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????n
NoOpNoOp&^ranking_model/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z:?????????:?????????:?????????:?????????: : : : : : : : : : : : : : : 2N
%ranking_model/StatefulPartitionedCall%ranking_model/StatefulPartitionedCall:L H
#
_output_shapes
:?????????
!
_user_specified_name	user_id:VR
#
_output_shapes
:?????????
+
_user_specified_namevideo_description:MI
#
_output_shapes
:?????????
"
_user_specified_name
video_id:QM
#
_output_shapes
:?????????
&
_user_specified_namevideo_rating:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
:
__inference__creator_15905
identity??
hash_tablek

hash_tableHashTableV2*
_output_shapes
: *
	key_dtype0	*
shared_name104*
value_dtype0	W
IdentityIdentityhash_table:table_handle:0^NoOp*
T0*
_output_shapes
: S
NoOpNoOp^hash_table*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 2

hash_table
hash_table
?
?
F__inference_video_model_layer_call_and_return_conditional_losses_15079
user_id	
video_description
video_id	
video_rating
ranking_model_15047
ranking_model_15049	%
ranking_model_15051:$ 
ranking_model_15053
ranking_model_15055	%
ranking_model_15057: 
ranking_model_15059
ranking_model_15061	&
ranking_model_15063:	? &
ranking_model_15065:	@?"
ranking_model_15067:	?&
ranking_model_15069:	?@!
ranking_model_15071:@%
ranking_model_15073:@!
ranking_model_15075:
identity??%ranking_model/StatefulPartitionedCall?
%ranking_model/StatefulPartitionedCallStatefulPartitionedCalluser_idvideo_idvideo_descriptionranking_model_15047ranking_model_15049ranking_model_15051ranking_model_15053ranking_model_15055ranking_model_15057ranking_model_15059ranking_model_15061ranking_model_15063ranking_model_15065ranking_model_15067ranking_model_15069ranking_model_15071ranking_model_15073ranking_model_15075*
Tin
2					*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*+
_read_only_resource_inputs
	*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_ranking_model_layer_call_and_return_conditional_losses_14541}
IdentityIdentity.ranking_model/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????n
NoOpNoOp&^ranking_model/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z:?????????:?????????:?????????:?????????: : : : : : : : : : : : : : : 2N
%ranking_model/StatefulPartitionedCall%ranking_model/StatefulPartitionedCall:L H
#
_output_shapes
:?????????
!
_user_specified_name	user_id:VR
#
_output_shapes
:?????????
+
_user_specified_namevideo_description:MI
#
_output_shapes
:?????????
"
_user_specified_name
video_id:QM
#
_output_shapes
:?????????
&
_user_specified_namevideo_rating:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
F__inference_embedding_1_layer_call_and_return_conditional_losses_14086

inputs	(
embedding_lookup_14080: 
identity??embedding_lookup?
embedding_lookupResourceGatherembedding_lookup_14080inputs*
Tindices0	*)
_class
loc:@embedding_lookup/14080*'
_output_shapes
:????????? *
dtype0?
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*)
_class
loc:@embedding_lookup/14080*'
_output_shapes
:????????? }
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:????????? s
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*'
_output_shapes
:????????? Y
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:?????????: 2$
embedding_lookupembedding_lookup:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
G__inference_sequential_3_layer_call_and_return_conditional_losses_14423

inputs
dense_14407:	@?
dense_14409:	? 
dense_1_14412:	?@
dense_1_14414:@
dense_2_14417:@
dense_2_14419:
identity??dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?dense_2/StatefulPartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCallinputsdense_14407dense_14409*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_14300?
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_14412dense_1_14414*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_14317?
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_14417dense_2_14419*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_14333w
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????@: : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
E__inference_sequential_layer_call_and_return_conditional_losses_14066
integer_lookup_input	=
9integer_lookup_none_lookup_lookuptablefindv2_table_handle>
:integer_lookup_none_lookup_lookuptablefindv2_default_value	!
embedding_14062:$ 
identity??!embedding/StatefulPartitionedCall?,integer_lookup/None_Lookup/LookupTableFindV2?
,integer_lookup/None_Lookup/LookupTableFindV2LookupTableFindV29integer_lookup_none_lookup_lookuptablefindv2_table_handleinteger_lookup_input:integer_lookup_none_lookup_lookuptablefindv2_default_value*	
Tin0	*

Tout0	*#
_output_shapes
:??????????
integer_lookup/IdentityIdentity5integer_lookup/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:??????????
!embedding/StatefulPartitionedCallStatefulPartitionedCall integer_lookup/Identity:output:0embedding_14062*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_embedding_layer_call_and_return_conditional_losses_13978y
IdentityIdentity*embedding/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:????????? ?
NoOpNoOp"^embedding/StatefulPartitionedCall-^integer_lookup/None_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????: : : 2F
!embedding/StatefulPartitionedCall!embedding/StatefulPartitionedCall2\
,integer_lookup/None_Lookup/LookupTableFindV2,integer_lookup/None_Lookup/LookupTableFindV2:Y U
#
_output_shapes
:?????????
.
_user_specified_nameinteger_lookup_input:

_output_shapes
: 
?P
?
H__inference_ranking_model_layer_call_and_return_conditional_losses_15476
inputs_0	
inputs_1	
inputs_2H
Dsequential_integer_lookup_none_lookup_lookuptablefindv2_table_handleI
Esequential_integer_lookup_none_lookup_lookuptablefindv2_default_value	=
+sequential_embedding_embedding_lookup_15430:$ I
Esequential_1_string_lookup_none_lookup_lookuptablefindv2_table_handleJ
Fsequential_1_string_lookup_none_lookup_lookuptablefindv2_default_value	A
/sequential_1_embedding_1_embedding_lookup_15439: L
Hsequential_2_integer_lookup_1_none_lookup_lookuptablefindv2_table_handleM
Isequential_2_integer_lookup_1_none_lookup_lookuptablefindv2_default_value	B
/sequential_2_embedding_2_embedding_lookup_15448:	? D
1sequential_3_dense_matmul_readvariableop_resource:	@?A
2sequential_3_dense_biasadd_readvariableop_resource:	?F
3sequential_3_dense_1_matmul_readvariableop_resource:	?@B
4sequential_3_dense_1_biasadd_readvariableop_resource:@E
3sequential_3_dense_2_matmul_readvariableop_resource:@B
4sequential_3_dense_2_biasadd_readvariableop_resource:
identity??%sequential/embedding/embedding_lookup?7sequential/integer_lookup/None_Lookup/LookupTableFindV2?)sequential_1/embedding_1/embedding_lookup?8sequential_1/string_lookup/None_Lookup/LookupTableFindV2?)sequential_2/embedding_2/embedding_lookup?;sequential_2/integer_lookup_1/None_Lookup/LookupTableFindV2?)sequential_3/dense/BiasAdd/ReadVariableOp?(sequential_3/dense/MatMul/ReadVariableOp?+sequential_3/dense_1/BiasAdd/ReadVariableOp?*sequential_3/dense_1/MatMul/ReadVariableOp?+sequential_3/dense_2/BiasAdd/ReadVariableOp?*sequential_3/dense_2/MatMul/ReadVariableOp?
7sequential/integer_lookup/None_Lookup/LookupTableFindV2LookupTableFindV2Dsequential_integer_lookup_none_lookup_lookuptablefindv2_table_handleinputs_0Esequential_integer_lookup_none_lookup_lookuptablefindv2_default_value*	
Tin0	*

Tout0	*#
_output_shapes
:??????????
"sequential/integer_lookup/IdentityIdentity@sequential/integer_lookup/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:??????????
%sequential/embedding/embedding_lookupResourceGather+sequential_embedding_embedding_lookup_15430+sequential/integer_lookup/Identity:output:0*
Tindices0	*>
_class4
20loc:@sequential/embedding/embedding_lookup/15430*'
_output_shapes
:????????? *
dtype0?
.sequential/embedding/embedding_lookup/IdentityIdentity.sequential/embedding/embedding_lookup:output:0*
T0*>
_class4
20loc:@sequential/embedding/embedding_lookup/15430*'
_output_shapes
:????????? ?
0sequential/embedding/embedding_lookup/Identity_1Identity7sequential/embedding/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:????????? ?
8sequential_1/string_lookup/None_Lookup/LookupTableFindV2LookupTableFindV2Esequential_1_string_lookup_none_lookup_lookuptablefindv2_table_handleinputs_2Fsequential_1_string_lookup_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
#sequential_1/string_lookup/IdentityIdentityAsequential_1/string_lookup/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:??????????
)sequential_1/embedding_1/embedding_lookupResourceGather/sequential_1_embedding_1_embedding_lookup_15439,sequential_1/string_lookup/Identity:output:0*
Tindices0	*B
_class8
64loc:@sequential_1/embedding_1/embedding_lookup/15439*'
_output_shapes
:????????? *
dtype0?
2sequential_1/embedding_1/embedding_lookup/IdentityIdentity2sequential_1/embedding_1/embedding_lookup:output:0*
T0*B
_class8
64loc:@sequential_1/embedding_1/embedding_lookup/15439*'
_output_shapes
:????????? ?
4sequential_1/embedding_1/embedding_lookup/Identity_1Identity;sequential_1/embedding_1/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:????????? ?
;sequential_2/integer_lookup_1/None_Lookup/LookupTableFindV2LookupTableFindV2Hsequential_2_integer_lookup_1_none_lookup_lookuptablefindv2_table_handleinputs_1Isequential_2_integer_lookup_1_none_lookup_lookuptablefindv2_default_value*	
Tin0	*

Tout0	*#
_output_shapes
:??????????
&sequential_2/integer_lookup_1/IdentityIdentityDsequential_2/integer_lookup_1/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:??????????
)sequential_2/embedding_2/embedding_lookupResourceGather/sequential_2_embedding_2_embedding_lookup_15448/sequential_2/integer_lookup_1/Identity:output:0*
Tindices0	*B
_class8
64loc:@sequential_2/embedding_2/embedding_lookup/15448*'
_output_shapes
:????????? *
dtype0?
2sequential_2/embedding_2/embedding_lookup/IdentityIdentity2sequential_2/embedding_2/embedding_lookup:output:0*
T0*B
_class8
64loc:@sequential_2/embedding_2/embedding_lookup/15448*'
_output_shapes
:????????? ?
4sequential_2/embedding_2/embedding_lookup/Identity_1Identity;sequential_2/embedding_2/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:????????? M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :?
concatConcatV29sequential/embedding/embedding_lookup/Identity_1:output:0=sequential_1/embedding_1/embedding_lookup/Identity_1:output:0concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????@?
(sequential_3/dense/MatMul/ReadVariableOpReadVariableOp1sequential_3_dense_matmul_readvariableop_resource*
_output_shapes
:	@?*
dtype0?
sequential_3/dense/MatMulMatMulconcat:output:00sequential_3/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
)sequential_3/dense/BiasAdd/ReadVariableOpReadVariableOp2sequential_3_dense_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
sequential_3/dense/BiasAddBiasAdd#sequential_3/dense/MatMul:product:01sequential_3/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????w
sequential_3/dense/ReluRelu#sequential_3/dense/BiasAdd:output:0*
T0*(
_output_shapes
:???????????
*sequential_3/dense_1/MatMul/ReadVariableOpReadVariableOp3sequential_3_dense_1_matmul_readvariableop_resource*
_output_shapes
:	?@*
dtype0?
sequential_3/dense_1/MatMulMatMul%sequential_3/dense/Relu:activations:02sequential_3/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
+sequential_3/dense_1/BiasAdd/ReadVariableOpReadVariableOp4sequential_3_dense_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
sequential_3/dense_1/BiasAddBiasAdd%sequential_3/dense_1/MatMul:product:03sequential_3/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@z
sequential_3/dense_1/ReluRelu%sequential_3/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@?
*sequential_3/dense_2/MatMul/ReadVariableOpReadVariableOp3sequential_3_dense_2_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0?
sequential_3/dense_2/MatMulMatMul'sequential_3/dense_1/Relu:activations:02sequential_3/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
+sequential_3/dense_2/BiasAdd/ReadVariableOpReadVariableOp4sequential_3_dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
sequential_3/dense_2/BiasAddBiasAdd%sequential_3/dense_2/MatMul:product:03sequential_3/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????t
IdentityIdentity%sequential_3/dense_2/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp&^sequential/embedding/embedding_lookup8^sequential/integer_lookup/None_Lookup/LookupTableFindV2*^sequential_1/embedding_1/embedding_lookup9^sequential_1/string_lookup/None_Lookup/LookupTableFindV2*^sequential_2/embedding_2/embedding_lookup<^sequential_2/integer_lookup_1/None_Lookup/LookupTableFindV2*^sequential_3/dense/BiasAdd/ReadVariableOp)^sequential_3/dense/MatMul/ReadVariableOp,^sequential_3/dense_1/BiasAdd/ReadVariableOp+^sequential_3/dense_1/MatMul/ReadVariableOp,^sequential_3/dense_2/BiasAdd/ReadVariableOp+^sequential_3/dense_2/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*^
_input_shapesM
K:?????????:?????????:?????????: : : : : : : : : : : : : : : 2N
%sequential/embedding/embedding_lookup%sequential/embedding/embedding_lookup2r
7sequential/integer_lookup/None_Lookup/LookupTableFindV27sequential/integer_lookup/None_Lookup/LookupTableFindV22V
)sequential_1/embedding_1/embedding_lookup)sequential_1/embedding_1/embedding_lookup2t
8sequential_1/string_lookup/None_Lookup/LookupTableFindV28sequential_1/string_lookup/None_Lookup/LookupTableFindV22V
)sequential_2/embedding_2/embedding_lookup)sequential_2/embedding_2/embedding_lookup2z
;sequential_2/integer_lookup_1/None_Lookup/LookupTableFindV2;sequential_2/integer_lookup_1/None_Lookup/LookupTableFindV22V
)sequential_3/dense/BiasAdd/ReadVariableOp)sequential_3/dense/BiasAdd/ReadVariableOp2T
(sequential_3/dense/MatMul/ReadVariableOp(sequential_3/dense/MatMul/ReadVariableOp2Z
+sequential_3/dense_1/BiasAdd/ReadVariableOp+sequential_3/dense_1/BiasAdd/ReadVariableOp2X
*sequential_3/dense_1/MatMul/ReadVariableOp*sequential_3/dense_1/MatMul/ReadVariableOp2Z
+sequential_3/dense_2/BiasAdd/ReadVariableOp+sequential_3/dense_2/BiasAdd/ReadVariableOp2X
*sequential_3/dense_2/MatMul/ReadVariableOp*sequential_3/dense_2/MatMul/ReadVariableOp:M I
#
_output_shapes
:?????????
"
_user_specified_name
inputs/0:MI
#
_output_shapes
:?????????
"
_user_specified_name
inputs/1:MI
#
_output_shapes
:?????????
"
_user_specified_name
inputs/2:

_output_shapes
: :

_output_shapes
: :


_output_shapes
: 
?
?
+__inference_video_model_layer_call_fn_15041
user_id	
video_description
video_id	
video_rating
unknown
	unknown_0	
	unknown_1:$ 
	unknown_2
	unknown_3	
	unknown_4: 
	unknown_5
	unknown_6	
	unknown_7:	? 
	unknown_8:	@?
	unknown_9:	?

unknown_10:	?@

unknown_11:@

unknown_12:@

unknown_13:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalluser_idvideo_descriptionvideo_idvideo_ratingunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13*
Tin
2					*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*+
_read_only_resource_inputs
		*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_video_model_layer_call_and_return_conditional_losses_14970o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z:?????????:?????????:?????????:?????????: : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:L H
#
_output_shapes
:?????????
!
_user_specified_name	user_id:VR
#
_output_shapes
:?????????
+
_user_specified_namevideo_description:MI
#
_output_shapes
:?????????
"
_user_specified_name
video_id:QM
#
_output_shapes
:?????????
&
_user_specified_namevideo_rating:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
,__inference_sequential_1_layer_call_fn_15601

inputs
unknown
	unknown_0	
	unknown_1: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_sequential_1_layer_call_and_return_conditional_losses_14132o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:????????? `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: 
?a
?
F__inference_video_model_layer_call_and_return_conditional_losses_15291
features_user_id	
features_video_description
features_video_id	
features_video_ratingV
Rranking_model_sequential_integer_lookup_none_lookup_lookuptablefindv2_table_handleW
Sranking_model_sequential_integer_lookup_none_lookup_lookuptablefindv2_default_value	K
9ranking_model_sequential_embedding_embedding_lookup_15245:$ W
Sranking_model_sequential_1_string_lookup_none_lookup_lookuptablefindv2_table_handleX
Tranking_model_sequential_1_string_lookup_none_lookup_lookuptablefindv2_default_value	O
=ranking_model_sequential_1_embedding_1_embedding_lookup_15254: Z
Vranking_model_sequential_2_integer_lookup_1_none_lookup_lookuptablefindv2_table_handle[
Wranking_model_sequential_2_integer_lookup_1_none_lookup_lookuptablefindv2_default_value	P
=ranking_model_sequential_2_embedding_2_embedding_lookup_15263:	? R
?ranking_model_sequential_3_dense_matmul_readvariableop_resource:	@?O
@ranking_model_sequential_3_dense_biasadd_readvariableop_resource:	?T
Aranking_model_sequential_3_dense_1_matmul_readvariableop_resource:	?@P
Branking_model_sequential_3_dense_1_biasadd_readvariableop_resource:@S
Aranking_model_sequential_3_dense_2_matmul_readvariableop_resource:@P
Branking_model_sequential_3_dense_2_biasadd_readvariableop_resource:
identity??3ranking_model/sequential/embedding/embedding_lookup?Eranking_model/sequential/integer_lookup/None_Lookup/LookupTableFindV2?7ranking_model/sequential_1/embedding_1/embedding_lookup?Franking_model/sequential_1/string_lookup/None_Lookup/LookupTableFindV2?7ranking_model/sequential_2/embedding_2/embedding_lookup?Iranking_model/sequential_2/integer_lookup_1/None_Lookup/LookupTableFindV2?7ranking_model/sequential_3/dense/BiasAdd/ReadVariableOp?6ranking_model/sequential_3/dense/MatMul/ReadVariableOp?9ranking_model/sequential_3/dense_1/BiasAdd/ReadVariableOp?8ranking_model/sequential_3/dense_1/MatMul/ReadVariableOp?9ranking_model/sequential_3/dense_2/BiasAdd/ReadVariableOp?8ranking_model/sequential_3/dense_2/MatMul/ReadVariableOp?
Eranking_model/sequential/integer_lookup/None_Lookup/LookupTableFindV2LookupTableFindV2Rranking_model_sequential_integer_lookup_none_lookup_lookuptablefindv2_table_handlefeatures_user_idSranking_model_sequential_integer_lookup_none_lookup_lookuptablefindv2_default_value*	
Tin0	*

Tout0	*#
_output_shapes
:??????????
0ranking_model/sequential/integer_lookup/IdentityIdentityNranking_model/sequential/integer_lookup/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:??????????
3ranking_model/sequential/embedding/embedding_lookupResourceGather9ranking_model_sequential_embedding_embedding_lookup_152459ranking_model/sequential/integer_lookup/Identity:output:0*
Tindices0	*L
_classB
@>loc:@ranking_model/sequential/embedding/embedding_lookup/15245*'
_output_shapes
:????????? *
dtype0?
<ranking_model/sequential/embedding/embedding_lookup/IdentityIdentity<ranking_model/sequential/embedding/embedding_lookup:output:0*
T0*L
_classB
@>loc:@ranking_model/sequential/embedding/embedding_lookup/15245*'
_output_shapes
:????????? ?
>ranking_model/sequential/embedding/embedding_lookup/Identity_1IdentityEranking_model/sequential/embedding/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:????????? ?
Franking_model/sequential_1/string_lookup/None_Lookup/LookupTableFindV2LookupTableFindV2Sranking_model_sequential_1_string_lookup_none_lookup_lookuptablefindv2_table_handlefeatures_video_descriptionTranking_model_sequential_1_string_lookup_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
1ranking_model/sequential_1/string_lookup/IdentityIdentityOranking_model/sequential_1/string_lookup/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:??????????
7ranking_model/sequential_1/embedding_1/embedding_lookupResourceGather=ranking_model_sequential_1_embedding_1_embedding_lookup_15254:ranking_model/sequential_1/string_lookup/Identity:output:0*
Tindices0	*P
_classF
DBloc:@ranking_model/sequential_1/embedding_1/embedding_lookup/15254*'
_output_shapes
:????????? *
dtype0?
@ranking_model/sequential_1/embedding_1/embedding_lookup/IdentityIdentity@ranking_model/sequential_1/embedding_1/embedding_lookup:output:0*
T0*P
_classF
DBloc:@ranking_model/sequential_1/embedding_1/embedding_lookup/15254*'
_output_shapes
:????????? ?
Branking_model/sequential_1/embedding_1/embedding_lookup/Identity_1IdentityIranking_model/sequential_1/embedding_1/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:????????? ?
Iranking_model/sequential_2/integer_lookup_1/None_Lookup/LookupTableFindV2LookupTableFindV2Vranking_model_sequential_2_integer_lookup_1_none_lookup_lookuptablefindv2_table_handlefeatures_video_idWranking_model_sequential_2_integer_lookup_1_none_lookup_lookuptablefindv2_default_value*	
Tin0	*

Tout0	*#
_output_shapes
:??????????
4ranking_model/sequential_2/integer_lookup_1/IdentityIdentityRranking_model/sequential_2/integer_lookup_1/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:??????????
7ranking_model/sequential_2/embedding_2/embedding_lookupResourceGather=ranking_model_sequential_2_embedding_2_embedding_lookup_15263=ranking_model/sequential_2/integer_lookup_1/Identity:output:0*
Tindices0	*P
_classF
DBloc:@ranking_model/sequential_2/embedding_2/embedding_lookup/15263*'
_output_shapes
:????????? *
dtype0?
@ranking_model/sequential_2/embedding_2/embedding_lookup/IdentityIdentity@ranking_model/sequential_2/embedding_2/embedding_lookup:output:0*
T0*P
_classF
DBloc:@ranking_model/sequential_2/embedding_2/embedding_lookup/15263*'
_output_shapes
:????????? ?
Branking_model/sequential_2/embedding_2/embedding_lookup/Identity_1IdentityIranking_model/sequential_2/embedding_2/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:????????? [
ranking_model/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :?
ranking_model/concatConcatV2Granking_model/sequential/embedding/embedding_lookup/Identity_1:output:0Kranking_model/sequential_1/embedding_1/embedding_lookup/Identity_1:output:0"ranking_model/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????@?
6ranking_model/sequential_3/dense/MatMul/ReadVariableOpReadVariableOp?ranking_model_sequential_3_dense_matmul_readvariableop_resource*
_output_shapes
:	@?*
dtype0?
'ranking_model/sequential_3/dense/MatMulMatMulranking_model/concat:output:0>ranking_model/sequential_3/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
7ranking_model/sequential_3/dense/BiasAdd/ReadVariableOpReadVariableOp@ranking_model_sequential_3_dense_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
(ranking_model/sequential_3/dense/BiasAddBiasAdd1ranking_model/sequential_3/dense/MatMul:product:0?ranking_model/sequential_3/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
%ranking_model/sequential_3/dense/ReluRelu1ranking_model/sequential_3/dense/BiasAdd:output:0*
T0*(
_output_shapes
:???????????
8ranking_model/sequential_3/dense_1/MatMul/ReadVariableOpReadVariableOpAranking_model_sequential_3_dense_1_matmul_readvariableop_resource*
_output_shapes
:	?@*
dtype0?
)ranking_model/sequential_3/dense_1/MatMulMatMul3ranking_model/sequential_3/dense/Relu:activations:0@ranking_model/sequential_3/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
9ranking_model/sequential_3/dense_1/BiasAdd/ReadVariableOpReadVariableOpBranking_model_sequential_3_dense_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
*ranking_model/sequential_3/dense_1/BiasAddBiasAdd3ranking_model/sequential_3/dense_1/MatMul:product:0Aranking_model/sequential_3/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
'ranking_model/sequential_3/dense_1/ReluRelu3ranking_model/sequential_3/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@?
8ranking_model/sequential_3/dense_2/MatMul/ReadVariableOpReadVariableOpAranking_model_sequential_3_dense_2_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0?
)ranking_model/sequential_3/dense_2/MatMulMatMul5ranking_model/sequential_3/dense_1/Relu:activations:0@ranking_model/sequential_3/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
9ranking_model/sequential_3/dense_2/BiasAdd/ReadVariableOpReadVariableOpBranking_model_sequential_3_dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
*ranking_model/sequential_3/dense_2/BiasAddBiasAdd3ranking_model/sequential_3/dense_2/MatMul:product:0Aranking_model/sequential_3/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
IdentityIdentity3ranking_model/sequential_3/dense_2/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp4^ranking_model/sequential/embedding/embedding_lookupF^ranking_model/sequential/integer_lookup/None_Lookup/LookupTableFindV28^ranking_model/sequential_1/embedding_1/embedding_lookupG^ranking_model/sequential_1/string_lookup/None_Lookup/LookupTableFindV28^ranking_model/sequential_2/embedding_2/embedding_lookupJ^ranking_model/sequential_2/integer_lookup_1/None_Lookup/LookupTableFindV28^ranking_model/sequential_3/dense/BiasAdd/ReadVariableOp7^ranking_model/sequential_3/dense/MatMul/ReadVariableOp:^ranking_model/sequential_3/dense_1/BiasAdd/ReadVariableOp9^ranking_model/sequential_3/dense_1/MatMul/ReadVariableOp:^ranking_model/sequential_3/dense_2/BiasAdd/ReadVariableOp9^ranking_model/sequential_3/dense_2/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z:?????????:?????????:?????????:?????????: : : : : : : : : : : : : : : 2j
3ranking_model/sequential/embedding/embedding_lookup3ranking_model/sequential/embedding/embedding_lookup2?
Eranking_model/sequential/integer_lookup/None_Lookup/LookupTableFindV2Eranking_model/sequential/integer_lookup/None_Lookup/LookupTableFindV22r
7ranking_model/sequential_1/embedding_1/embedding_lookup7ranking_model/sequential_1/embedding_1/embedding_lookup2?
Franking_model/sequential_1/string_lookup/None_Lookup/LookupTableFindV2Franking_model/sequential_1/string_lookup/None_Lookup/LookupTableFindV22r
7ranking_model/sequential_2/embedding_2/embedding_lookup7ranking_model/sequential_2/embedding_2/embedding_lookup2?
Iranking_model/sequential_2/integer_lookup_1/None_Lookup/LookupTableFindV2Iranking_model/sequential_2/integer_lookup_1/None_Lookup/LookupTableFindV22r
7ranking_model/sequential_3/dense/BiasAdd/ReadVariableOp7ranking_model/sequential_3/dense/BiasAdd/ReadVariableOp2p
6ranking_model/sequential_3/dense/MatMul/ReadVariableOp6ranking_model/sequential_3/dense/MatMul/ReadVariableOp2v
9ranking_model/sequential_3/dense_1/BiasAdd/ReadVariableOp9ranking_model/sequential_3/dense_1/BiasAdd/ReadVariableOp2t
8ranking_model/sequential_3/dense_1/MatMul/ReadVariableOp8ranking_model/sequential_3/dense_1/MatMul/ReadVariableOp2v
9ranking_model/sequential_3/dense_2/BiasAdd/ReadVariableOp9ranking_model/sequential_3/dense_2/BiasAdd/ReadVariableOp2t
8ranking_model/sequential_3/dense_2/MatMul/ReadVariableOp8ranking_model/sequential_3/dense_2/MatMul/ReadVariableOp:U Q
#
_output_shapes
:?????????
*
_user_specified_namefeatures/user_id:_[
#
_output_shapes
:?????????
4
_user_specified_namefeatures/video_description:VR
#
_output_shapes
:?????????
+
_user_specified_namefeatures/video_id:ZV
#
_output_shapes
:?????????
/
_user_specified_namefeatures/video_rating:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
,__inference_sequential_2_layer_call_fn_14260
integer_lookup_1_input	
unknown
	unknown_0	
	unknown_1:	? 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinteger_lookup_1_inputunknown	unknown_0	unknown_1*
Tin
2		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_sequential_2_layer_call_and_return_conditional_losses_14240o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:????????? `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
#
_output_shapes
:?????????
0
_user_specified_nameinteger_lookup_1_input:

_output_shapes
: 
?4
?

__inference__traced_save_16025
file_prefix3
/savev2_embedding_embeddings_read_readvariableop5
1savev2_embedding_1_embeddings_read_readvariableop5
1savev2_embedding_2_embeddings_read_readvariableop+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop-
)savev2_dense_1_kernel_read_readvariableop+
'savev2_dense_1_bias_read_readvariableop-
)savev2_dense_2_kernel_read_readvariableop+
'savev2_dense_2_bias_read_readvariableop+
'savev2_adagrad_iter_read_readvariableop	,
(savev2_adagrad_decay_read_readvariableop4
0savev2_adagrad_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableopG
Csavev2_adagrad_embedding_embeddings_accumulator_read_readvariableopI
Esavev2_adagrad_embedding_1_embeddings_accumulator_read_readvariableop?
;savev2_adagrad_dense_kernel_accumulator_read_readvariableop=
9savev2_adagrad_dense_bias_accumulator_read_readvariableopA
=savev2_adagrad_dense_1_kernel_accumulator_read_readvariableop?
;savev2_adagrad_dense_1_bias_accumulator_read_readvariableopA
=savev2_adagrad_dense_2_kernel_accumulator_read_readvariableop?
;savev2_adagrad_dense_2_bias_accumulator_read_readvariableop
savev2_const_9

identity_1??MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : ?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ?

SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?

value?	B?	B&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBLvariables/0/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEBLvariables/1/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEBLvariables/3/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEBLvariables/4/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEBLvariables/5/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEBLvariables/6/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEBLvariables/7/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEBLvariables/8/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*A
value8B6B B B B B B B B B B B B B B B B B B B B B B B ?

SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0/savev2_embedding_embeddings_read_readvariableop1savev2_embedding_1_embeddings_read_readvariableop1savev2_embedding_2_embeddings_read_readvariableop'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop)savev2_dense_2_kernel_read_readvariableop'savev2_dense_2_bias_read_readvariableop'savev2_adagrad_iter_read_readvariableop(savev2_adagrad_decay_read_readvariableop0savev2_adagrad_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableopCsavev2_adagrad_embedding_embeddings_accumulator_read_readvariableopEsavev2_adagrad_embedding_1_embeddings_accumulator_read_readvariableop;savev2_adagrad_dense_kernel_accumulator_read_readvariableop9savev2_adagrad_dense_bias_accumulator_read_readvariableop=savev2_adagrad_dense_1_kernel_accumulator_read_readvariableop;savev2_adagrad_dense_1_bias_accumulator_read_readvariableop=savev2_adagrad_dense_2_kernel_accumulator_read_readvariableop;savev2_adagrad_dense_2_bias_accumulator_read_readvariableopsavev2_const_9"/device:CPU:0*
_output_shapes
 *%
dtypes
2	?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*?
_input_shapes?
?: :$ : :	? :	@?:?:	?@:@:@:: : : : : :$ : :	@?:?:	?@:@:@:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:$ :$ 

_output_shapes

: :%!

_output_shapes
:	? :%!

_output_shapes
:	@?:!

_output_shapes	
:?:%!

_output_shapes
:	?@: 

_output_shapes
:@:$ 

_output_shapes

:@: 	

_output_shapes
::


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:$ :$ 

_output_shapes

: :%!

_output_shapes
:	@?:!

_output_shapes	
:?:%!

_output_shapes
:	?@: 

_output_shapes
:@:$ 

_output_shapes

:@: 

_output_shapes
::

_output_shapes
: 
?
?
F__inference_video_model_layer_call_and_return_conditional_losses_14970
features	

features_1

features_2	

features_3
ranking_model_14938
ranking_model_14940	%
ranking_model_14942:$ 
ranking_model_14944
ranking_model_14946	%
ranking_model_14948: 
ranking_model_14950
ranking_model_14952	&
ranking_model_14954:	? &
ranking_model_14956:	@?"
ranking_model_14958:	?&
ranking_model_14960:	?@!
ranking_model_14962:@%
ranking_model_14964:@!
ranking_model_14966:
identity??%ranking_model/StatefulPartitionedCall?
%ranking_model/StatefulPartitionedCallStatefulPartitionedCallfeatures
features_2
features_1ranking_model_14938ranking_model_14940ranking_model_14942ranking_model_14944ranking_model_14946ranking_model_14948ranking_model_14950ranking_model_14952ranking_model_14954ranking_model_14956ranking_model_14958ranking_model_14960ranking_model_14962ranking_model_14964ranking_model_14966*
Tin
2					*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*+
_read_only_resource_inputs
	*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_ranking_model_layer_call_and_return_conditional_losses_14657}
IdentityIdentity.ranking_model/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????n
NoOpNoOp&^ranking_model/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z:?????????:?????????:?????????:?????????: : : : : : : : : : : : : : : 2N
%ranking_model/StatefulPartitionedCall%ranking_model/StatefulPartitionedCall:M I
#
_output_shapes
:?????????
"
_user_specified_name
features:MI
#
_output_shapes
:?????????
"
_user_specified_name
features:MI
#
_output_shapes
:?????????
"
_user_specified_name
features:MI
#
_output_shapes
:?????????
"
_user_specified_name
features:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: "?	L
saver_filename:0StatefulPartitionedCall_4:0StatefulPartitionedCall_58"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
7
user_id,
serving_default_user_id:0	?????????
K
video_description6
#serving_default_video_description:0?????????
9
video_id-
serving_default_video_id:0	?????????
A
video_rating1
serving_default_video_rating:0?????????<
output_10
StatefulPartitionedCall:0?????????tensorflow/serving/predict:??
?
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
ranking_model
	task

	optimizer
loss

signatures"
_tf_keras_model
_
0
1
2
3
4
5
6
7
8"
trackable_list_wrapper
_
0
1
2
3
4
5
6
7
8"
trackable_list_wrapper
 "
trackable_list_wrapper
?
non_trainable_variables

layers
metrics
layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
?
trace_0
trace_1
trace_2
trace_32?
+__inference_video_model_layer_call_fn_14889
+__inference_video_model_layer_call_fn_15197
+__inference_video_model_layer_call_fn_15235
+__inference_video_model_layer_call_fn_15041?
???
FullArgSpec
args?
jself

jfeatures
varargs
 
varkw
 
defaults
 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 ztrace_0ztrace_1ztrace_2ztrace_3
?
trace_0
 trace_1
!trace_2
"trace_32?
F__inference_video_model_layer_call_and_return_conditional_losses_15291
F__inference_video_model_layer_call_and_return_conditional_losses_15347
F__inference_video_model_layer_call_and_return_conditional_losses_15079
F__inference_video_model_layer_call_and_return_conditional_losses_15117?
???
FullArgSpec
args?
jself

jfeatures
varargs
 
varkw
 
defaults
 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 ztrace_0z trace_1z!trace_2z"trace_3
?
#	capture_1
$	capture_4
%	capture_7B?
 __inference__wrapped_model_13958user_idvideo_descriptionvideo_idvideo_rating"?
???
FullArgSpec
args? 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z#	capture_1z$	capture_4z%	capture_7
?
&	variables
'trainable_variables
(regularization_losses
)	keras_api
*__call__
*+&call_and_return_all_conditional_losses
,user_embeddings
 -video_description_embeddings
.video_id_embeddings
/ratings"
_tf_keras_model
?
0	variables
1trainable_variables
2regularization_losses
3	keras_api
4__call__
*5&call_and_return_all_conditional_losses
6_ranking_metrics
7_prediction_metrics
8_label_metrics
9_loss_metrics"
_tf_keras_layer
?
:iter
	;decay
<learning_rateaccumulator?accumulator?accumulator?accumulator?accumulator?accumulator?accumulator?accumulator?"
	optimizer
 "
trackable_dict_wrapper
,
=serving_default"
signature_map
&:$$ 2embedding/embeddings
(:& 2embedding_1/embeddings
):'	? 2embedding_2/embeddings
:	@?2dense/kernel
:?2
dense/bias
!:	?@2dense_1/kernel
:@2dense_1/bias
 :@2dense_2/kernel
:2dense_2/bias
 "
trackable_list_wrapper
.
0
	1"
trackable_list_wrapper
'
>0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?
#	capture_1
$	capture_4
%	capture_7B?
+__inference_video_model_layer_call_fn_14889user_idvideo_descriptionvideo_idvideo_rating"?
???
FullArgSpec
args?
jself

jfeatures
varargs
 
varkw
 
defaults
 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 z#	capture_1z$	capture_4z%	capture_7
?
#	capture_1
$	capture_4
%	capture_7B?
+__inference_video_model_layer_call_fn_15197features/user_idfeatures/video_descriptionfeatures/video_idfeatures/video_rating"?
???
FullArgSpec
args?
jself

jfeatures
varargs
 
varkw
 
defaults
 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 z#	capture_1z$	capture_4z%	capture_7
?
#	capture_1
$	capture_4
%	capture_7B?
+__inference_video_model_layer_call_fn_15235features/user_idfeatures/video_descriptionfeatures/video_idfeatures/video_rating"?
???
FullArgSpec
args?
jself

jfeatures
varargs
 
varkw
 
defaults
 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 z#	capture_1z$	capture_4z%	capture_7
?
#	capture_1
$	capture_4
%	capture_7B?
+__inference_video_model_layer_call_fn_15041user_idvideo_descriptionvideo_idvideo_rating"?
???
FullArgSpec
args?
jself

jfeatures
varargs
 
varkw
 
defaults
 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 z#	capture_1z$	capture_4z%	capture_7
?
#	capture_1
$	capture_4
%	capture_7B?
F__inference_video_model_layer_call_and_return_conditional_losses_15291features/user_idfeatures/video_descriptionfeatures/video_idfeatures/video_rating"?
???
FullArgSpec
args?
jself

jfeatures
varargs
 
varkw
 
defaults
 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 z#	capture_1z$	capture_4z%	capture_7
?
#	capture_1
$	capture_4
%	capture_7B?
F__inference_video_model_layer_call_and_return_conditional_losses_15347features/user_idfeatures/video_descriptionfeatures/video_idfeatures/video_rating"?
???
FullArgSpec
args?
jself

jfeatures
varargs
 
varkw
 
defaults
 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 z#	capture_1z$	capture_4z%	capture_7
?
#	capture_1
$	capture_4
%	capture_7B?
F__inference_video_model_layer_call_and_return_conditional_losses_15079user_idvideo_descriptionvideo_idvideo_rating"?
???
FullArgSpec
args?
jself

jfeatures
varargs
 
varkw
 
defaults
 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 z#	capture_1z$	capture_4z%	capture_7
?
#	capture_1
$	capture_4
%	capture_7B?
F__inference_video_model_layer_call_and_return_conditional_losses_15117user_idvideo_descriptionvideo_idvideo_rating"?
???
FullArgSpec
args?
jself

jfeatures
varargs
 
varkw
 
defaults
 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 z#	capture_1z$	capture_4z%	capture_7
!J	
Const_8jtf.TrackableConstant
!J	
Const_7jtf.TrackableConstant
!J	
Const_6jtf.TrackableConstant
_
0
1
2
3
4
5
6
7
8"
trackable_list_wrapper
_
0
1
2
3
4
5
6
7
8"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables

@layers
Ametrics
Blayer_regularization_losses
Clayer_metrics
&	variables
'trainable_variables
(regularization_losses
*__call__
*+&call_and_return_all_conditional_losses
&+"call_and_return_conditional_losses"
_generic_user_object
?
Dtrace_0
Etrace_1
Ftrace_2
Gtrace_32?
-__inference_ranking_model_layer_call_fn_14574
-__inference_ranking_model_layer_call_fn_15384
-__inference_ranking_model_layer_call_fn_15421
-__inference_ranking_model_layer_call_fn_14727?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 zDtrace_0zEtrace_1zFtrace_2zGtrace_3
?
Htrace_0
Itrace_1
Jtrace_2
Ktrace_32?
H__inference_ranking_model_layer_call_and_return_conditional_losses_15476
H__inference_ranking_model_layer_call_and_return_conditional_losses_15531
H__inference_ranking_model_layer_call_and_return_conditional_losses_14769
H__inference_ranking_model_layer_call_and_return_conditional_losses_14811?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 zHtrace_0zItrace_1zJtrace_2zKtrace_3
?
Llayer-0
Mlayer_with_weights-0
Mlayer-1
N	variables
Otrainable_variables
Pregularization_losses
Q	keras_api
R__call__
*S&call_and_return_all_conditional_losses"
_tf_keras_sequential
?
Tlayer-0
Ulayer_with_weights-0
Ulayer-1
V	variables
Wtrainable_variables
Xregularization_losses
Y	keras_api
Z__call__
*[&call_and_return_all_conditional_losses"
_tf_keras_sequential
?
\layer-0
]layer_with_weights-0
]layer-1
^	variables
_trainable_variables
`regularization_losses
a	keras_api
b__call__
*c&call_and_return_all_conditional_losses"
_tf_keras_sequential
?
dlayer_with_weights-0
dlayer-0
elayer_with_weights-1
elayer-1
flayer_with_weights-2
flayer-2
g	variables
htrainable_variables
iregularization_losses
j	keras_api
k__call__
*l&call_and_return_all_conditional_losses"
_tf_keras_sequential
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
mnon_trainable_variables

nlayers
ometrics
player_regularization_losses
qlayer_metrics
0	variables
1trainable_variables
2regularization_losses
4__call__
*5&call_and_return_all_conditional_losses
&5"call_and_return_conditional_losses"
_generic_user_object
?2??
???
FullArgSpec\
argsT?Q
jself
jlabels
jpredictions
jsample_weight

jtraining
jcompute_metrics
varargs
 
varkw
 
defaults?

 
p 
p

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec\
argsT?Q
jself
jlabels
jpredictions
jsample_weight

jtraining
jcompute_metrics
varargs
 
varkw
 
defaults?

 
p 
p

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
'
>0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
:	 (2Adagrad/iter
: (2Adagrad/decay
: (2Adagrad/learning_rate
?
#	capture_1
$	capture_4
%	capture_7B?
#__inference_signature_wrapper_15159user_idvideo_descriptionvideo_idvideo_rating"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z#	capture_1z$	capture_4z%	capture_7
N
r	variables
s	keras_api
	ttotal
	ucount"
_tf_keras_metric
 "
trackable_list_wrapper
<
,0
-1
.2
/3"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?
#	capture_1
$	capture_4
%	capture_7B?
-__inference_ranking_model_layer_call_fn_14574input_1input_2input_3"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 z#	capture_1z$	capture_4z%	capture_7
?
#	capture_1
$	capture_4
%	capture_7B?
-__inference_ranking_model_layer_call_fn_15384inputs/0inputs/1inputs/2"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 z#	capture_1z$	capture_4z%	capture_7
?
#	capture_1
$	capture_4
%	capture_7B?
-__inference_ranking_model_layer_call_fn_15421inputs/0inputs/1inputs/2"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 z#	capture_1z$	capture_4z%	capture_7
?
#	capture_1
$	capture_4
%	capture_7B?
-__inference_ranking_model_layer_call_fn_14727input_1input_2input_3"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 z#	capture_1z$	capture_4z%	capture_7
?
#	capture_1
$	capture_4
%	capture_7B?
H__inference_ranking_model_layer_call_and_return_conditional_losses_15476inputs/0inputs/1inputs/2"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 z#	capture_1z$	capture_4z%	capture_7
?
#	capture_1
$	capture_4
%	capture_7B?
H__inference_ranking_model_layer_call_and_return_conditional_losses_15531inputs/0inputs/1inputs/2"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 z#	capture_1z$	capture_4z%	capture_7
?
#	capture_1
$	capture_4
%	capture_7B?
H__inference_ranking_model_layer_call_and_return_conditional_losses_14769input_1input_2input_3"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 z#	capture_1z$	capture_4z%	capture_7
?
#	capture_1
$	capture_4
%	capture_7B?
H__inference_ranking_model_layer_call_and_return_conditional_losses_14811input_1input_2input_3"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 z#	capture_1z$	capture_4z%	capture_7
:
v	keras_api
wlookup_table"
_tf_keras_layer
?
x	variables
ytrainable_variables
zregularization_losses
{	keras_api
|__call__
*}&call_and_return_all_conditional_losses

embeddings"
_tf_keras_layer
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
?
~non_trainable_variables

layers
?metrics
 ?layer_regularization_losses
?layer_metrics
N	variables
Otrainable_variables
Pregularization_losses
R__call__
*S&call_and_return_all_conditional_losses
&S"call_and_return_conditional_losses"
_generic_user_object
?
?trace_0
?trace_1
?trace_2
?trace_32?
*__inference_sequential_layer_call_fn_13992
*__inference_sequential_layer_call_fn_15542
*__inference_sequential_layer_call_fn_15553
*__inference_sequential_layer_call_fn_14044?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0z?trace_1z?trace_2z?trace_3
?
?trace_0
?trace_1
?trace_2
?trace_32?
E__inference_sequential_layer_call_and_return_conditional_losses_15566
E__inference_sequential_layer_call_and_return_conditional_losses_15579
E__inference_sequential_layer_call_and_return_conditional_losses_14055
E__inference_sequential_layer_call_and_return_conditional_losses_14066?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0z?trace_1z?trace_2z?trace_3
<
?	keras_api
?lookup_table"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses

embeddings"
_tf_keras_layer
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
V	variables
Wtrainable_variables
Xregularization_losses
Z__call__
*[&call_and_return_all_conditional_losses
&["call_and_return_conditional_losses"
_generic_user_object
?
?trace_0
?trace_1
?trace_2
?trace_32?
,__inference_sequential_1_layer_call_fn_14100
,__inference_sequential_1_layer_call_fn_15590
,__inference_sequential_1_layer_call_fn_15601
,__inference_sequential_1_layer_call_fn_14152?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0z?trace_1z?trace_2z?trace_3
?
?trace_0
?trace_1
?trace_2
?trace_32?
G__inference_sequential_1_layer_call_and_return_conditional_losses_15614
G__inference_sequential_1_layer_call_and_return_conditional_losses_15627
G__inference_sequential_1_layer_call_and_return_conditional_losses_14163
G__inference_sequential_1_layer_call_and_return_conditional_losses_14174?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0z?trace_1z?trace_2z?trace_3
<
?	keras_api
?lookup_table"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses

embeddings"
_tf_keras_layer
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
^	variables
_trainable_variables
`regularization_losses
b__call__
*c&call_and_return_all_conditional_losses
&c"call_and_return_conditional_losses"
_generic_user_object
?
?trace_0
?trace_1
?trace_2
?trace_32?
,__inference_sequential_2_layer_call_fn_14208
,__inference_sequential_2_layer_call_fn_15638
,__inference_sequential_2_layer_call_fn_15649
,__inference_sequential_2_layer_call_fn_14260?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0z?trace_1z?trace_2z?trace_3
?
?trace_0
?trace_1
?trace_2
?trace_32?
G__inference_sequential_2_layer_call_and_return_conditional_losses_15662
G__inference_sequential_2_layer_call_and_return_conditional_losses_15675
G__inference_sequential_2_layer_call_and_return_conditional_losses_14271
G__inference_sequential_2_layer_call_and_return_conditional_losses_14282?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0z?trace_1z?trace_2z?trace_3
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
J
0
1
2
3
4
5"
trackable_list_wrapper
J
0
1
2
3
4
5"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
g	variables
htrainable_variables
iregularization_losses
k__call__
*l&call_and_return_all_conditional_losses
&l"call_and_return_conditional_losses"
_generic_user_object
?
?trace_0
?trace_1
?trace_2
?trace_32?
,__inference_sequential_3_layer_call_fn_14355
,__inference_sequential_3_layer_call_fn_15692
,__inference_sequential_3_layer_call_fn_15709
,__inference_sequential_3_layer_call_fn_14455?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0z?trace_1z?trace_2z?trace_3
?
?trace_0
?trace_1
?trace_2
?trace_32?
G__inference_sequential_3_layer_call_and_return_conditional_losses_15733
G__inference_sequential_3_layer_call_and_return_conditional_losses_15757
G__inference_sequential_3_layer_call_and_return_conditional_losses_14474
G__inference_sequential_3_layer_call_and_return_conditional_losses_14493?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0z?trace_1z?trace_2z?trace_3
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
>0"
trackable_list_wrapper
 "
trackable_list_wrapper
=
>root_mean_squared_error"
trackable_dict_wrapper
.
t0
u1"
trackable_list_wrapper
-
r	variables"
_generic_user_object
:  (2total
:  (2count
"
_generic_user_object
j
?_initializer
?_create_resource
?_initialize
?_destroy_resourceR jtf.StaticHashTable
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
x	variables
ytrainable_variables
zregularization_losses
|__call__
*}&call_and_return_all_conditional_losses
&}"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
)__inference_embedding_layer_call_fn_15764?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
D__inference_embedding_layer_call_and_return_conditional_losses_15773?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
 "
trackable_list_wrapper
.
L0
M1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?
#	capture_1B?
*__inference_sequential_layer_call_fn_13992integer_lookup_input"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z#	capture_1
?
#	capture_1B?
*__inference_sequential_layer_call_fn_15542inputs"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z#	capture_1
?
#	capture_1B?
*__inference_sequential_layer_call_fn_15553inputs"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z#	capture_1
?
#	capture_1B?
*__inference_sequential_layer_call_fn_14044integer_lookup_input"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z#	capture_1
?
#	capture_1B?
E__inference_sequential_layer_call_and_return_conditional_losses_15566inputs"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z#	capture_1
?
#	capture_1B?
E__inference_sequential_layer_call_and_return_conditional_losses_15579inputs"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z#	capture_1
?
#	capture_1B?
E__inference_sequential_layer_call_and_return_conditional_losses_14055integer_lookup_input"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z#	capture_1
?
#	capture_1B?
E__inference_sequential_layer_call_and_return_conditional_losses_14066integer_lookup_input"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z#	capture_1
"
_generic_user_object
j
?_initializer
?_create_resource
?_initialize
?_destroy_resourceR jtf.StaticHashTable
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
+__inference_embedding_1_layer_call_fn_15780?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
F__inference_embedding_1_layer_call_and_return_conditional_losses_15789?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
 "
trackable_list_wrapper
.
T0
U1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?
$	capture_1B?
,__inference_sequential_1_layer_call_fn_14100string_lookup_input"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z$	capture_1
?
$	capture_1B?
,__inference_sequential_1_layer_call_fn_15590inputs"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z$	capture_1
?
$	capture_1B?
,__inference_sequential_1_layer_call_fn_15601inputs"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z$	capture_1
?
$	capture_1B?
,__inference_sequential_1_layer_call_fn_14152string_lookup_input"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z$	capture_1
?
$	capture_1B?
G__inference_sequential_1_layer_call_and_return_conditional_losses_15614inputs"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z$	capture_1
?
$	capture_1B?
G__inference_sequential_1_layer_call_and_return_conditional_losses_15627inputs"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z$	capture_1
?
$	capture_1B?
G__inference_sequential_1_layer_call_and_return_conditional_losses_14163string_lookup_input"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z$	capture_1
?
$	capture_1B?
G__inference_sequential_1_layer_call_and_return_conditional_losses_14174string_lookup_input"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z$	capture_1
"
_generic_user_object
j
?_initializer
?_create_resource
?_initialize
?_destroy_resourceR jtf.StaticHashTable
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
+__inference_embedding_2_layer_call_fn_15796?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
F__inference_embedding_2_layer_call_and_return_conditional_losses_15805?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
 "
trackable_list_wrapper
.
\0
]1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?
%	capture_1B?
,__inference_sequential_2_layer_call_fn_14208integer_lookup_1_input"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z%	capture_1
?
%	capture_1B?
,__inference_sequential_2_layer_call_fn_15638inputs"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z%	capture_1
?
%	capture_1B?
,__inference_sequential_2_layer_call_fn_15649inputs"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z%	capture_1
?
%	capture_1B?
,__inference_sequential_2_layer_call_fn_14260integer_lookup_1_input"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z%	capture_1
?
%	capture_1B?
G__inference_sequential_2_layer_call_and_return_conditional_losses_15662inputs"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z%	capture_1
?
%	capture_1B?
G__inference_sequential_2_layer_call_and_return_conditional_losses_15675inputs"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z%	capture_1
?
%	capture_1B?
G__inference_sequential_2_layer_call_and_return_conditional_losses_14271integer_lookup_1_input"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z%	capture_1
?
%	capture_1B?
G__inference_sequential_2_layer_call_and_return_conditional_losses_14282integer_lookup_1_input"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z%	capture_1
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
%__inference_dense_layer_call_fn_15814?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
@__inference_dense_layer_call_and_return_conditional_losses_15825?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
'__inference_dense_1_layer_call_fn_15834?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
B__inference_dense_1_layer_call_and_return_conditional_losses_15845?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
'__inference_dense_2_layer_call_fn_15854?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
B__inference_dense_2_layer_call_and_return_conditional_losses_15864?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
 "
trackable_list_wrapper
5
d0
e1
f2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
,__inference_sequential_3_layer_call_fn_14355dense_input"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
,__inference_sequential_3_layer_call_fn_15692inputs"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
,__inference_sequential_3_layer_call_fn_15709inputs"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
,__inference_sequential_3_layer_call_fn_14455dense_input"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
G__inference_sequential_3_layer_call_and_return_conditional_losses_15733inputs"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
G__inference_sequential_3_layer_call_and_return_conditional_losses_15757inputs"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
G__inference_sequential_3_layer_call_and_return_conditional_losses_14474dense_input"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
G__inference_sequential_3_layer_call_and_return_conditional_losses_14493dense_input"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
"
_generic_user_object
?
?trace_02?
__inference__creator_15869?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? z?trace_0
?
?trace_02?
__inference__initializer_15877?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? z?trace_0
?
?trace_02?
__inference__destroyer_15882?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? z?trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
)__inference_embedding_layer_call_fn_15764inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
D__inference_embedding_layer_call_and_return_conditional_losses_15773inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
"
_generic_user_object
?
?trace_02?
__inference__creator_15887?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? z?trace_0
?
?trace_02?
__inference__initializer_15895?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? z?trace_0
?
?trace_02?
__inference__destroyer_15900?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? z?trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
+__inference_embedding_1_layer_call_fn_15780inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
F__inference_embedding_1_layer_call_and_return_conditional_losses_15789inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
"
_generic_user_object
?
?trace_02?
__inference__creator_15905?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? z?trace_0
?
?trace_02?
__inference__initializer_15913?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? z?trace_0
?
?trace_02?
__inference__destroyer_15918?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? z?trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
+__inference_embedding_2_layer_call_fn_15796inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
F__inference_embedding_2_layer_call_and_return_conditional_losses_15805inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
%__inference_dense_layer_call_fn_15814inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
@__inference_dense_layer_call_and_return_conditional_losses_15825inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
'__inference_dense_1_layer_call_fn_15834inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
B__inference_dense_1_layer_call_and_return_conditional_losses_15845inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
'__inference_dense_2_layer_call_fn_15854inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
B__inference_dense_2_layer_call_and_return_conditional_losses_15864inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
__inference__creator_15869"?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?
?	capture_1
?	capture_2B?
__inference__initializer_15877"?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? z?	capture_1z?	capture_2
?B?
__inference__destroyer_15882"?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?B?
__inference__creator_15887"?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?
?	capture_1
?	capture_2B?
__inference__initializer_15895"?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? z?	capture_1z?	capture_2
?B?
__inference__destroyer_15900"?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?B?
__inference__creator_15905"?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?
?	capture_1
?	capture_2B?
__inference__initializer_15913"?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? z?	capture_1z?	capture_2
?B?
__inference__destroyer_15918"?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
!J	
Const_5jtf.TrackableConstant
!J	
Const_4jtf.TrackableConstant
!J	
Const_3jtf.TrackableConstant
!J	
Const_2jtf.TrackableConstant
J
Constjtf.TrackableConstant
!J	
Const_1jtf.TrackableConstant
8:6$ 2(Adagrad/embedding/embeddings/accumulator
::8 2*Adagrad/embedding_1/embeddings/accumulator
1:/	@?2 Adagrad/dense/kernel/accumulator
+:)?2Adagrad/dense/bias/accumulator
3:1	?@2"Adagrad/dense_1/kernel/accumulator
,:*@2 Adagrad/dense_1/bias/accumulator
2:0@2"Adagrad/dense_2/kernel/accumulator
,:*2 Adagrad/dense_2/bias/accumulator6
__inference__creator_15869?

? 
? "? 6
__inference__creator_15887?

? 
? "? 6
__inference__creator_15905?

? 
? "? 8
__inference__destroyer_15882?

? 
? "? 8
__inference__destroyer_15900?

? 
? "? 8
__inference__destroyer_15918?

? 
? "? A
__inference__initializer_15877w???

? 
? "? B
__inference__initializer_15895 ????

? 
? "? B
__inference__initializer_15913 ????

? 
? "? ?
 __inference__wrapped_model_13958?w#?$?%???
???
???
(
user_id?
user_id?????????	
<
video_description'?$
video_description?????????
*
video_id?
video_id?????????	
2
video_rating"?
video_rating?????????
? "3?0
.
output_1"?
output_1??????????
B__inference_dense_1_layer_call_and_return_conditional_losses_15845]0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????@
? {
'__inference_dense_1_layer_call_fn_15834P0?-
&?#
!?
inputs??????????
? "??????????@?
B__inference_dense_2_layer_call_and_return_conditional_losses_15864\/?,
%?"
 ?
inputs?????????@
? "%?"
?
0?????????
? z
'__inference_dense_2_layer_call_fn_15854O/?,
%?"
 ?
inputs?????????@
? "???????????
@__inference_dense_layer_call_and_return_conditional_losses_15825]/?,
%?"
 ?
inputs?????????@
? "&?#
?
0??????????
? y
%__inference_dense_layer_call_fn_15814P/?,
%?"
 ?
inputs?????????@
? "????????????
F__inference_embedding_1_layer_call_and_return_conditional_losses_15789W+?(
!?
?
inputs?????????	
? "%?"
?
0????????? 
? y
+__inference_embedding_1_layer_call_fn_15780J+?(
!?
?
inputs?????????	
? "?????????? ?
F__inference_embedding_2_layer_call_and_return_conditional_losses_15805W+?(
!?
?
inputs?????????	
? "%?"
?
0????????? 
? y
+__inference_embedding_2_layer_call_fn_15796J+?(
!?
?
inputs?????????	
? "?????????? ?
D__inference_embedding_layer_call_and_return_conditional_losses_15773W+?(
!?
?
inputs?????????	
? "%?"
?
0????????? 
? w
)__inference_embedding_layer_call_fn_15764J+?(
!?
?
inputs?????????	
? "?????????? ?
H__inference_ranking_model_layer_call_and_return_conditional_losses_14769?w#?$?%?|
e?b
`?]
?
input_1?????????	
?
input_2?????????	
?
input_3?????????
?

trainingp "%?"
?
0?????????
? ?
H__inference_ranking_model_layer_call_and_return_conditional_losses_14811?w#?$?%?|
e?b
`?]
?
input_1?????????	
?
input_2?????????	
?
input_3?????????
?

trainingp"%?"
?
0?????????
? ?
H__inference_ranking_model_layer_call_and_return_conditional_losses_15476?w#?$?%??
h?e
c?`
?
inputs/0?????????	
?
inputs/1?????????	
?
inputs/2?????????
?

trainingp "%?"
?
0?????????
? ?
H__inference_ranking_model_layer_call_and_return_conditional_losses_15531?w#?$?%??
h?e
c?`
?
inputs/0?????????	
?
inputs/1?????????	
?
inputs/2?????????
?

trainingp"%?"
?
0?????????
? ?
-__inference_ranking_model_layer_call_fn_14574?w#?$?%?|
e?b
`?]
?
input_1?????????	
?
input_2?????????	
?
input_3?????????
?

trainingp "???????????
-__inference_ranking_model_layer_call_fn_14727?w#?$?%?|
e?b
`?]
?
input_1?????????	
?
input_2?????????	
?
input_3?????????
?

trainingp"???????????
-__inference_ranking_model_layer_call_fn_15384?w#?$?%??
h?e
c?`
?
inputs/0?????????	
?
inputs/1?????????	
?
inputs/2?????????
?

trainingp "???????????
-__inference_ranking_model_layer_call_fn_15421?w#?$?%??
h?e
c?`
?
inputs/0?????????	
?
inputs/1?????????	
?
inputs/2?????????
?

trainingp"???????????
G__inference_sequential_1_layer_call_and_return_conditional_losses_14163o?$@?=
6?3
)?&
string_lookup_input?????????
p 

 
? "%?"
?
0????????? 
? ?
G__inference_sequential_1_layer_call_and_return_conditional_losses_14174o?$@?=
6?3
)?&
string_lookup_input?????????
p

 
? "%?"
?
0????????? 
? ?
G__inference_sequential_1_layer_call_and_return_conditional_losses_15614b?$3?0
)?&
?
inputs?????????
p 

 
? "%?"
?
0????????? 
? ?
G__inference_sequential_1_layer_call_and_return_conditional_losses_15627b?$3?0
)?&
?
inputs?????????
p

 
? "%?"
?
0????????? 
? ?
,__inference_sequential_1_layer_call_fn_14100b?$@?=
6?3
)?&
string_lookup_input?????????
p 

 
? "?????????? ?
,__inference_sequential_1_layer_call_fn_14152b?$@?=
6?3
)?&
string_lookup_input?????????
p

 
? "?????????? ?
,__inference_sequential_1_layer_call_fn_15590U?$3?0
)?&
?
inputs?????????
p 

 
? "?????????? ?
,__inference_sequential_1_layer_call_fn_15601U?$3?0
)?&
?
inputs?????????
p

 
? "?????????? ?
G__inference_sequential_2_layer_call_and_return_conditional_losses_14271r?%C?@
9?6
,?)
integer_lookup_1_input?????????	
p 

 
? "%?"
?
0????????? 
? ?
G__inference_sequential_2_layer_call_and_return_conditional_losses_14282r?%C?@
9?6
,?)
integer_lookup_1_input?????????	
p

 
? "%?"
?
0????????? 
? ?
G__inference_sequential_2_layer_call_and_return_conditional_losses_15662b?%3?0
)?&
?
inputs?????????	
p 

 
? "%?"
?
0????????? 
? ?
G__inference_sequential_2_layer_call_and_return_conditional_losses_15675b?%3?0
)?&
?
inputs?????????	
p

 
? "%?"
?
0????????? 
? ?
,__inference_sequential_2_layer_call_fn_14208e?%C?@
9?6
,?)
integer_lookup_1_input?????????	
p 

 
? "?????????? ?
,__inference_sequential_2_layer_call_fn_14260e?%C?@
9?6
,?)
integer_lookup_1_input?????????	
p

 
? "?????????? ?
,__inference_sequential_2_layer_call_fn_15638U?%3?0
)?&
?
inputs?????????	
p 

 
? "?????????? ?
,__inference_sequential_2_layer_call_fn_15649U?%3?0
)?&
?
inputs?????????	
p

 
? "?????????? ?
G__inference_sequential_3_layer_call_and_return_conditional_losses_14474m<?9
2?/
%?"
dense_input?????????@
p 

 
? "%?"
?
0?????????
? ?
G__inference_sequential_3_layer_call_and_return_conditional_losses_14493m<?9
2?/
%?"
dense_input?????????@
p

 
? "%?"
?
0?????????
? ?
G__inference_sequential_3_layer_call_and_return_conditional_losses_15733h7?4
-?*
 ?
inputs?????????@
p 

 
? "%?"
?
0?????????
? ?
G__inference_sequential_3_layer_call_and_return_conditional_losses_15757h7?4
-?*
 ?
inputs?????????@
p

 
? "%?"
?
0?????????
? ?
,__inference_sequential_3_layer_call_fn_14355`<?9
2?/
%?"
dense_input?????????@
p 

 
? "???????????
,__inference_sequential_3_layer_call_fn_14455`<?9
2?/
%?"
dense_input?????????@
p

 
? "???????????
,__inference_sequential_3_layer_call_fn_15692[7?4
-?*
 ?
inputs?????????@
p 

 
? "???????????
,__inference_sequential_3_layer_call_fn_15709[7?4
-?*
 ?
inputs?????????@
p

 
? "???????????
E__inference_sequential_layer_call_and_return_conditional_losses_14055ow#A?>
7?4
*?'
integer_lookup_input?????????	
p 

 
? "%?"
?
0????????? 
? ?
E__inference_sequential_layer_call_and_return_conditional_losses_14066ow#A?>
7?4
*?'
integer_lookup_input?????????	
p

 
? "%?"
?
0????????? 
? ?
E__inference_sequential_layer_call_and_return_conditional_losses_15566aw#3?0
)?&
?
inputs?????????	
p 

 
? "%?"
?
0????????? 
? ?
E__inference_sequential_layer_call_and_return_conditional_losses_15579aw#3?0
)?&
?
inputs?????????	
p

 
? "%?"
?
0????????? 
? ?
*__inference_sequential_layer_call_fn_13992bw#A?>
7?4
*?'
integer_lookup_input?????????	
p 

 
? "?????????? ?
*__inference_sequential_layer_call_fn_14044bw#A?>
7?4
*?'
integer_lookup_input?????????	
p

 
? "?????????? ?
*__inference_sequential_layer_call_fn_15542Tw#3?0
)?&
?
inputs?????????	
p 

 
? "?????????? ?
*__inference_sequential_layer_call_fn_15553Tw#3?0
)?&
?
inputs?????????	
p

 
? "?????????? ?
#__inference_signature_wrapper_15159?w#?$?%???
? 
???
(
user_id?
user_id?????????	
<
video_description'?$
video_description?????????
*
video_id?
video_id?????????	
2
video_rating"?
video_rating?????????"3?0
.
output_1"?
output_1??????????
F__inference_video_model_layer_call_and_return_conditional_losses_15079?w#?$?%???
???
???
(
user_id?
user_id?????????	
<
video_description'?$
video_description?????????
*
video_id?
video_id?????????	
2
video_rating"?
video_rating?????????
?

trainingp "%?"
?
0?????????
? ?
F__inference_video_model_layer_call_and_return_conditional_losses_15117?w#?$?%???
???
???
(
user_id?
user_id?????????	
<
video_description'?$
video_description?????????
*
video_id?
video_id?????????	
2
video_rating"?
video_rating?????????
?

trainingp"%?"
?
0?????????
? ?
F__inference_video_model_layer_call_and_return_conditional_losses_15291?w#?$?%???
???
???
1
user_id&?#
features/user_id?????????	
E
video_description0?-
features/video_description?????????
3
video_id'?$
features/video_id?????????	
;
video_rating+?(
features/video_rating?????????
?

trainingp "%?"
?
0?????????
? ?
F__inference_video_model_layer_call_and_return_conditional_losses_15347?w#?$?%???
???
???
1
user_id&?#
features/user_id?????????	
E
video_description0?-
features/video_description?????????
3
video_id'?$
features/video_id?????????	
;
video_rating+?(
features/video_rating?????????
?

trainingp"%?"
?
0?????????
? ?
+__inference_video_model_layer_call_fn_14889?w#?$?%???
???
???
(
user_id?
user_id?????????	
<
video_description'?$
video_description?????????
*
video_id?
video_id?????????	
2
video_rating"?
video_rating?????????
?

trainingp "???????????
+__inference_video_model_layer_call_fn_15041?w#?$?%???
???
???
(
user_id?
user_id?????????	
<
video_description'?$
video_description?????????
*
video_id?
video_id?????????	
2
video_rating"?
video_rating?????????
?

trainingp"???????????
+__inference_video_model_layer_call_fn_15197?w#?$?%???
???
???
1
user_id&?#
features/user_id?????????	
E
video_description0?-
features/video_description?????????
3
video_id'?$
features/video_id?????????	
;
video_rating+?(
features/video_rating?????????
?

trainingp "???????????
+__inference_video_model_layer_call_fn_15235?w#?$?%???
???
???
1
user_id&?#
features/user_id?????????	
E
video_description0?-
features/video_description?????????
3
video_id'?$
features/video_id?????????	
;
video_rating+?(
features/video_rating?????????
?

trainingp"??????????