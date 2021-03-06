# Changelog
All important changes to the project are documented in this file.
## V1.0
First code version includes SMP processor which returns relevant quantities such as:
- R(H) plots;
- Magnetoresistance value;
- Resistance at the desired field;
- Group multiple SMPs R(H) plots by the index of measurement.
## V2.0
The second version includes SMP processor and data visualizer.
- In addition to the previous functions the data processor also includes:
  - The low plateau resistance value.
  - Open and short circuit threshold for filtering undesired measurements;
  - A SMP progress bar and a function progress bar are now included.
- The data visualizer can plot all the relevant quantities on an interactive plot saved under .HTML format.
## V3.0
The third version includes SMP processor and data visualizer.
- In addition to the previous functions the data processor also includes:
  - A text field that displays what is being done when the code is running;
  - The "R at Field [Oe]:" entry field is removed and if users asks for that function a pop-up window shows up asking the desired field;
  - The messages field was deleted;
  - Pop-up messages were introduced.
- The data visualizer had the following updates:
  - The messages field was deleted;
  - Pop-up messages were introduced.
## V3.1
Minor buggs fixed, this is the final version of the data processor and data visualizer, next step was to include an automatic curve classifier based on ML.
## V4.0
Automatic curve classifier based on ML included. Classifier built for TMR based sensors.

The V4.0 is available on the respository.
