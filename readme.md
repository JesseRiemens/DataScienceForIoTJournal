# Person Detector using Tflite, raspberrypi camera and Homebridge
### N.B.: ik zie nu dat de code die in python staat nog grotendeels de originele comments (en header) heeft van Evan Juras. Mijn code is een modificatie op zijn modificatie van https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/examples/python/label_image.py, en heeft veel toegvoegde elementen. Het originele project is te zien op https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10.
Nodig: 
 - Raspberry pi 3b/4
 - Benodidge libraries in venv beschreven in https://www.digikey.com/en/maker/projects/how-to-perform-object-detection-with-tensorflow-lite-on-raspberry-pi/b929e1519c7c43d5b2c6f89984883588
 - Een werkende Homebridge Instance 
 - Een AppleTV (of andere home hub) die dienst doet als gateway voor de HB instance
 - Een internetverbinding (optioneel voor externe toegang)

 ## Het idee
 De Raspberry Pi Camera vangt een beeld op, wat geprocessed wordt met het TFlite model in python. Dit stuurt een "confidence" uit met de zekerheid dat het object dat gezien is een mens is. Wanneer deze zekerheid over de drempelwaarde heengaat wordt het frame opgeslagen in `./programoutput/detectedHuman.jpg`. Een andere thread maakt deze foto zichtbaar op een webpagina. (`localhost:50505/detectedHuman.jpg`) Deze foto wordt door de homebridge-server gelezen d.m.v. de [homebridge-camera-ffmpeg](https://github.com/Sunoo/homebridge-camera-ffmpeg) plugin. Homebridge haalt zo af en toe een snapshot op van de webpagina, en stuurt deze naar de AppleTV (of een andere Homekit Hub). Deze hub stuurt vervolgens de camerabeelden naar de clients die de beelden opvragen.
 
 ## Voordelen
- Set up and forget, er zijn geen abbonementskosten.
- Open source.
- Veilig. Er is geen handgemaakte verbinding naar het internet, en geen open/geforwarde poorten nodig.
- Schaalbaar. Tegen ongeveer EUR50 heb je een detector.
- Aanpasbaar. Wanneer je een eigen model wil inladen om alleen bepaalde mensen wel of niet te detecteren is dat mogelijk
- Geintegreerd. Doordat het werkt met Apple Homekit hoeven er geen nieuwe apps gedownload te worden en werkt het naadloos samen met andere IoT-appliances. 

## Nadelen
- Het opzetten van een werkende TFLite omgeving is redelijk tijdrovend (zie TODO Docker)
- Er is een homebridge server nodig. (Maar laten we eerlijk zijn, als je eigen IoT devices bouwt voor je Apple Ecosysteem heb je die waarschijnlijk toch al) Deze server kan ook op de Pi 4 draaien.
- Een Pi 4 gebruikt redelijk wat stroom, dus een accu-implementatie is niet mogelijk. Daarvoor zou een geconverteerd model (lees: met int's ipv floats) gemaakt moeten worden en geimplementeerd moeten worden op iets als een ESP32. Hierdoor moet de Pi 4 altijd aan een lichtnet-adapter hangen of met een PoE-HAT worden gevoed.
- Voor bewegingsdetectie is een continue video-stream nodig, en een MQTT server op de Homebridge Server.
- Momenteel wordt niet het meest efficiente TF-model gebruikt. Het huidige model kan meer dan alleen mensen detecteren, en een ideaal model zou alleen gefocust zijn op mensen, en eventueel specifieke mensen detecteren. Dit is geen probleem, omdat de functie nu goed is. Een eigen model trainen zou helaas extreem veel tijd (en geld) kosten. Bron: TinyML van O'Reilly
## TODO's
### Een MQTT-melding wanneer er een nieuwe detectie is
Deze MQTT-melding kan worden doorgegeven aan de Homebridge-instance om in iOS een deurbel-melding te geven wanneer er iemand is gedetecteerd. Hiervoor is een MQTT-server nodig en extra implementatie in de config van homebridge, naast een MQTT-implementatie in python. 
### Een video-stream naast de image-capture
Dit is te implementeren maar blijkt lastiger dan het zou lijken. Hiervoor is een server als op [deze](https://www.pyimagesearch.com/2019/09/02/opencv-stream-video-to-web-browser-html-page/) pagina nodig, en dit is een redelijk groot project. Dit zou te groot zijn voor de scope van dit keuzevak.
### Docker
Een Docker-container zou het proces van setup erg vergemakkelijken. Dit is alleen in deze scope iets te veel gevraagd.
## Video
Een video van de werking is te zien in deze video: [![Testvideo](https://img.youtube.com/vi/URSCkwpq7DU/0.jpg)](https://youtube.com/shorts/URSCkwpq7DU?feature=share)

Verdere video's verschijnen later vandaag in deze playlist: [playlist](https://youtube.com/playlist?list=PLm1u9so5js4u3rSxmQwpRpVraKscDPqEZ)
