# The add-points form follows the October GeoJSON columns

The screen is "Add points". Latitude and longitude are required dot-decimal floats in the geographic range, typed in a text field so there is no spinner. TIME is a required choice of a project age (65 to 145 Ma, step 5, including 145 Ma, which the viewer hides). Climate_Cl is a required Dry / Semi-arid / Humid choice. Country, basin, formation and the other identifying columns sit with those. The rest of the October columns, except Weight, sit under Optional. Empty optional text is null. Show_on_Ma, when set, is one of those ages as an integer. PlateID, when set, is an integer.

ANCHOR, FILE1, RECONFILE1 and ID are not on the form. The download sets ANCHOR to 0, FILE1 to the filename `point-YYYYMMDD-HHMMSS.geojson`, RECONFILE1 to `Zahirovic_etal_2022_OptimisedMantleRef_and_NNRMantleRef.rot`, and ID to null. The October table already has null ids, and the screen was not given a rule for minting one.

**Considered options:** (1) this export; (2) a timestamp integer for ID; (3) leave FILE1 and RECONFILE1 as fields the author types. (2) invents an id the table did not ask for. (3) asks for two values that are fixed for every new row.
