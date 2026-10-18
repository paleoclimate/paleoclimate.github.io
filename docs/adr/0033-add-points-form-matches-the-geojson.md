# The add-points form follows the October GeoJSON columns

Superseded in part by `docs/adr/0036-show-on-ma-copies-time.md` and `docs/adr/0037-first-section-fields-are-required.md`. Show_on_Ma is no longer a field, and Group, Member, and Well name/outcrop are optional. The required section is the list in 0037.

The screen is "Add points". Latitude and longitude are required dot-decimal floats in the geographic range, typed in a text field so there is no spinner. TIME is a required choice of a project age (65 to 145 Ma, step 5, including 145 Ma). Climate_Cl is a required Dry / Semi-arid / Humid choice. Country, basin, formation, PlateID and REF(Authors, Year) are required with those. Empty optional text is null. PlateID is an integer.

ANCHOR, FILE1, RECONFILE1 and ID are not on the form. The download sets ANCHOR to 0, FILE1 to the filename `point-YYYYMMDD-HHMMSS.geojson`, RECONFILE1 to `Zahirovic_etal_2022_OptimisedMantleRef_and_NNRMantleRef.rot`, and ID to null. The October table already has null ids, and the screen was not given a rule for minting one.

**Considered options:** (1) this export; (2) a timestamp integer for ID; (3) leave FILE1 and RECONFILE1 as fields the author types. (2) invents an id the table did not ask for. (3) asks for two values that are fixed for every new row.
