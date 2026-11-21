# Hospital integration cookbook

Two research teams need to read the same hospital notes. Both can work without
patient names. One can also do without dates, while the other has approval to
use visit dates to reconstruct a timeline. Running the model over every note
is the expensive part. Can the hospital do that work once and still give each
team the text its study needs?

## Follow one note from the hospital record to the study

Take this fictional Dutch note.

> Patiënt Alex Peters kwam op 03/03/2026.

In this example, Databricks runs the hospital's data-processing job. The job
reads one revision of the note and sends its exact text to MedDeID over the
hospital's private network. The model runs on a GPU. It can identify
`Alex Peters` as a patient name and `03/03/2026` as a date. For each detection
it returns the label, the matched text and its position in the original note.
Together these make up a **span**.

The original note stays in the hospital's document store. In a restricted
table, the job saves the positions and labels with the document revision they
belong to. It can then read that revision and apply each study's rules to the
marked words. MedDeID calls this step **rendering**. The study team receives
the resulting text.

| What the study is allowed to see | Text from the same two detections |
| --- | --- |
| A note with names and dates masked | `Patiënt [Name:Patient] kwam op [Date].` |
| The visit date for a timeline review | `Patiënt [Name:Patient] kwam op 03/03/2026.` |

## When do we need the GPU again?

Suppose the timeline study gets approval after the first dataset has been
delivered. The job can make the second text on a CPU from the saved spans and
the same source revision. The GPU can stay off.

Now suppose a clinician adds a sentence to the note. An old position might
point into the middle of a different word, and the new sentence might name
another person. The job sends the corrected revision through detection before
using it in a study. A change to the model or to the patient details used to
recognize names also calls for fresh detections.

## Set this up at your hospital

The worked example uses Azure, one NVIDIA T4 and a Belgian Dutch model.
Databricks reads the hospital's versioned documents and writes the results to
Delta tables. Each team can start with its part.

1. **Release owner.** Give IT the approved software and model, and give the
   data team a way to check which model answered its request.
   [Prepare the release](deploy-service.md#choose-the-release){ #1-prepare-a-release }
2. **Hospital IT.** Give the job a private HTTPS address where it can send a
   note, and check that only authorized callers get through.
   [Set up the host](deploy-service.md#prepare-a-private-gpu-host){ #2-provision-the-host-with-terraform }
   · [Start the HTTPS service](deploy-service.md#start-the-https-service){ #3-start-the-service-behind-nginx }
3. **Data platform team.** Try the connection with the fictional note above.
   Then use the hospital's own records to produce and release an approved study
   column. [Test one note](databricks-job.md#try-one-fictional-note){ #4-run-one-synthetic-note-from-databricks }
   · [Use versioned documents](databricks-job.md#read-the-revision-that-can-be-used-for-the-study){ #5-connect-the-job-to-versioned-documents }

For a study that needs consistent aliases in notes and table columns, see
[Render the study text](databricks-job.md#render-the-study-text){ #request-one-rendered-text-or-reuse-detections-locally }.
Before the first clinical batch, [check the installation on local notes and test recovery](databricks-job.md#before-the-first-clinical-batch){ #6-accept-and-operate-the-installation }.
