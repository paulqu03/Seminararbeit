Dieses Repository enthält den Code, der im Rahmen meiner Seminararbeit entwickelt wurde.

Falls man die Textextraktion mit den ersten beiden Programmen durchführen möchte (sehr rechenaufwändig), muss erst noch die Videos herunterladen (abspeichern in der Form video_0, video_1,...), oder kann andere Videos nutzen (in dem Fall müssen aber noch Anpassungen im Code vorgenommen werden)
Die Videos findet man unter: https://www.dropbox.com/scl/fo/efw4ut6eue1mox35d5lgh/AIPAGjRUA68aQlPOzNQvnao/videos?dl=0&e=11&rlkey=sxzt5cdio1fuyxk4rew9ur11g&subfolder_nav_tracking=1
Danach musss man die Videos noch einzeln per Command umwandeln mit folgenden Command, damit alles funktioniert:

ffmpeg -i video_10.mp4 -c:v libx264 -preset fast -crf 22 -c:a aac -b:a 128k video_10_h264.mp4


Ich konnte aus Zeitgründen bei der Shap Analyse vom Bart Modell nur ein Label untersuchen. Auch hier müssen Änderungen aam Code vorgenommen werden, wenn man ein anderes Label untersuchen möchte.


Ich empfehle mit 2 Venv zu arbeiten, wobei einss nur für easyOCR genutzt wird.
Das aufsetzen der Enviromnents kann sich je nach PC unterscheiden, je nachdem ob man eine CUDA-fähige Grafikkarte hatoder nicht.
Außerdem empfehle ich das 4. Programm in mehreren Schritten zu bearbeiten, da es ssehr rechenaufwändig ist.


