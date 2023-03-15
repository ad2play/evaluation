# Ad2Play - Evaluation

## Files

 - **data/digital_twins.json** structures both the Siemens and Cisco digital twins. 
 - **data/pattern.json** defines relevant matching terms.
 - **data/csaf.json** stores the converted CSAF documents.
 - **data/cacao.json** holds the generated CACAO playbooks.
 
 - **evaluation.csv** provides the summary for the evaluation.

## Imports


```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pywaffle import Waffle
import plotly.express as px
import matplotlib.patches as mpatches

from matplotlib import rc
```

## User Definitions


```python
sns.set(font_scale=1.4)
sentences_siemens = 220
sentences_cisa = 705
sentences_cisco = 136
sentences = sentences_siemens + sentences_cisa + sentences_cisco

def plot_confusion_matrix(source, df_tasks, sentences):
    fp = df_tasks.loc[df_tasks['source'] == source, 'useless_steps'].sum()
    fn = df_tasks.loc[df_tasks['source'] == source, 'actions_missing'].sum()
    tp = df_tasks.loc[df_tasks['source'] == source, 'detected_actions'].sum() - fp - df_tasks.loc[df_tasks['source'] == source, 'matched_wrong_pattern'].sum() - df_tasks.loc[df_tasks['source'] == source, 'nlp_error'].sum()
    tn = sentences - fp - fn - tp

    accuracy = (tp+tn)/(tp+fn+tn+fp)*100
    precision = tp/(tp+fp)*100
    recall = tp/(tp+fn)*100
    f1_score = (2 * precision * recall)/(precision + recall)

    ax = sns.heatmap([[tp,fp],[fn,tn]],cbar=False, annot=True, cmap='binary', fmt='.4g', alpha=0.7)

    stats = "\n\nAccuracy = {:0.2f}%        Precision = {:0.2f}%\nRecall = {:0.2f}%             F1 Score = {:0.2f}%".format(
                    accuracy,precision,recall,f1_score)
    ax.text(0.0, 2.5, stats,fontsize=17)
    ax.set_xlabel('Actual Actions\n', fontdict=dict(weight='bold'))
    ax.set_ylabel('Predicted Actions\n', fontdict=dict(weight='bold'))

    values = ["(TP)", "(FP)", "(FN)", "(TN)"]
    ax.texts[0].set_text(ax.texts[0].get_text() + "\n" + values[0])
    ax.texts[1].set_text(ax.texts[1].get_text() + "\n" + values[1])
    ax.texts[2].set_text(ax.texts[2].get_text() + "\n" + values[2])
    ax.texts[3].set_text(ax.texts[3].get_text() + "\n" + values[3])


    ## Ticket labels - List must be in alphabetical order
    ax.xaxis.set_ticklabels(['Positive','Negative'])
    ax.yaxis.set_ticklabels(['Positive','Negative'])
    ax.xaxis.set_label_position('top')
    ax.xaxis.tick_top()
    ax.yaxis.set_label_position('left')
    ax.yaxis.tick_left()

    plt.savefig(f'./charts/confusion_matrix_{source}.pdf', dpi=200, bbox_inches="tight")
    plt.show()

```

## Import CSV


```python
df_tasks = pd.read_csv("evaluation.csv", delimiter=";")
df_tasks.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>task_id</th>
      <th>csaf_id</th>
      <th>playbook_id</th>
      <th>twin</th>
      <th>source</th>
      <th>manual</th>
      <th>detected_actions</th>
      <th>actual_actions</th>
      <th>actions_missing</th>
      <th>missing_actions</th>
      <th>...</th>
      <th>update_step</th>
      <th>investigation_step</th>
      <th>locating_step</th>
      <th>data_operation_step</th>
      <th>isolation_step</th>
      <th>access_action_step</th>
      <th>system_action_step</th>
      <th>set_entity_step</th>
      <th>traffic_action_step</th>
      <th>observe_behavior_step</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>62c3f0af99cf253386589061</td>
      <td>62c3f0af99cf253386588bb0</td>
      <td>62c44b404466fa24127b09c9</td>
      <td>Ad2Play:Mock_Siemens</td>
      <td>Siemens ProductCERT</td>
      <td>False</td>
      <td>5</td>
      <td>5</td>
      <td>0</td>
      <td>NaN</td>
      <td>...</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>62c3f0af99cf253386588bae</td>
      <td>62c3f0af99cf253386588945</td>
      <td>62c44b404466fa24127b0858</td>
      <td>Ad2Play:Mock_Siemens</td>
      <td>Siemens ProductCERT</td>
      <td>False</td>
      <td>2</td>
      <td>2</td>
      <td>0</td>
      <td>NaN</td>
      <td>...</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>62c3f0af99cf253386588943</td>
      <td>62c3f0af99cf253386588527</td>
      <td>62c44b404466fa24127b079d</td>
      <td>Ad2Play:Mock_Siemens</td>
      <td>Siemens ProductCERT</td>
      <td>False</td>
      <td>3</td>
      <td>3</td>
      <td>0</td>
      <td>NaN</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>62c3f0af99cf253386588525</td>
      <td>62c3f0af99cf253386588485</td>
      <td>62c44b3f4466fa24127b0652</td>
      <td>Ad2Play:Mock_Siemens</td>
      <td>Siemens ProductCERT</td>
      <td>False</td>
      <td>4</td>
      <td>4</td>
      <td>0</td>
      <td>NaN</td>
      <td>...</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>62c3f0af99cf253386588483</td>
      <td>62c3f0af99cf2533865882e7</td>
      <td>62c44b3f4466fa24127b05f6</td>
      <td>Ad2Play:Mock_Siemens</td>
      <td>Siemens ProductCERT</td>
      <td>False</td>
      <td>5</td>
      <td>8</td>
      <td>3</td>
      <td>Update to V4.8 HF6;  Deactivate the webserver ...</td>
      <td>...</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 25 columns</p>
</div>



## CERT Quality

### Advisories and Actions by CERT


```python
# Use TeX for rendering text
rc('text', usetex=True)

# Extract actions and advisories for each source
cisco_actions = df_tasks.loc[df_tasks['source'] == "Cisco Security Advisory", 'detected_actions'].sum()
cisa_actions = df_tasks.loc[df_tasks['source'] == "CISA ICS CERT", 'detected_actions'].sum()
siemens_actions = df_tasks.loc[df_tasks['source'] == "Siemens ProductCERT", 'detected_actions'].sum()
cisco_advisories = df_tasks.loc[df_tasks['source'] == "Cisco Security Advisory", 'source'].count()
cisa_advisories = df_tasks.loc[df_tasks['source'] == "CISA ICS CERT", 'source'].count()
siemens_advisories = df_tasks.loc[df_tasks['source'] == "Siemens ProductCERT", 'source'].count()

# Create lists for pie charts
advisories = [siemens_advisories, cisa_advisories, cisco_advisories]
actions = [siemens_actions, cisa_actions, cisco_actions]

# Define labels and colors
mylabels = ["Siemens ProductCERT", "CISA ICS CERT", "Cisco CERT"]
mycolors = ["#555555", "#BBBBBB", "#beb9db"]

# Define autopct format for pie charts
def autopct_format(values):
    def my_format(pct):
        total = sum(values)
        val = int(round(pct*total/100.0))
        return '{v:d}'.format(v=val)
    return my_format

# Create subplots
fig, axs = plt.subplots(2, figsize=(4,7))

# Plot the first pie chart
axs[0].pie(advisories, autopct = autopct_format(advisories), pctdistance=1.15, colors = mycolors)
axs[0].set_title(r"\textbf{Advisories}")

# Plot the second pie chart
axs[1].pie(actions, autopct = autopct_format(actions), pctdistance=1.15, colors = mycolors)
axs[1].set_title(r"\textbf{Actions}")

# Add a legend
legend = plt.legend(mylabels, bbox_to_anchor=(0.9, 0.1), fontsize=17, 
                    bbox_transform=plt.gcf().transFigure, frameon=False)
legend.get_frame().set_alpha(None)
legend.get_frame().set_facecolor((0, 0, 0, 0))

# Show the plot
plt.show()

# Save the plot to a file
fig.savefig('./charts/actions.pdf', dpi = 200, bbox_inches="tight")

```


    
![png](output_11_0.png)
    


### Categorization of Actions


```python
import matplotlib.pyplot as plt
from pywaffle import Waffle

# Define data
data = {
    "Update": df_tasks["update_step"].sum(),
    "Investigation": df_tasks["investigation_step"].sum(),
    "Locating": df_tasks["locating_step"].sum(),
    "Data-Operation": df_tasks["data_operation_step"].sum(),
    "Isolation": df_tasks["isolation_step"].sum(),
    "Privileges": df_tasks["access_action_step"].sum(),
    "System": df_tasks["system_action_step"].sum(),
    "Configuration": df_tasks["set_entity_step"].sum(),
    "Network": df_tasks["traffic_action_step"].sum(),
    "Observation": df_tasks["observe_behavior_step"].sum(),
}

# Sort data by descending order of values
data = dict(sorted(data.items(), key=lambda x: x[1], reverse=True))

# Define chart parameters
nRows = 15
colors = ["#BBBBBB", "#7eb0d5",  "#8bd3c7", "#bd7ebe", "#ffb55a", "#555555", "#beb9db", "#fdcce5",  "#888888","#b2e061"]

# Calculate total sum of values
total = sum(data.values())

# Create a list of legend labels with percentages
legend_labels = []
for k, v in data.items():
    percent = (v / total) * 100
    if percent != 0:
        legend_labels.append(f"{k} ({percent:.1f}%)")
    else:
        legend_labels.append(f"{k} ({percent:.0f}%)")

# Create the Waffle chart with modified legend labels
fig = plt.figure(
    FigureClass=Waffle,
    rows=nRows,
    colors=colors,
    figsize=(16, 17),
    values=data,
    legend={
        "loc": "upper left",
        "bbox_to_anchor": (0, -0.05),
        "ncol": 5,
        "framealpha": 0,
        "labels": legend_labels,
        "prop": {"weight": "bold"},
    },
)

# Show the chart
plt.show()

# Save the chart to a file
plt.rc("text", usetex=False)
fig.savefig("./charts/action_types.pdf", dpi=200, bbox_inches="tight")

```


    
![png](output_13_0.png)
    


## Playbook Quality

### Confusion Matrix for Siemens ProductCERT


```python
plot_confusion_matrix("Siemens ProductCERT", df_tasks, sentences_siemens)
```


    
![png](output_16_0.png)
    


### Confusion Matrix for Cisco Security Advisory


```python
plot_confusion_matrix('Cisco Security Advisory', df_tasks, sentences_siemens)
```


    
![png](output_18_0.png)
    


### Confusion Matrix for CISA ICS CERT


```python
plot_confusion_matrix("CISA ICS CERT", df_tasks, sentences_cisa)
```


    
![png](output_20_0.png)
    


### Confusion Matrix CERT Summary


```python
# Calculate the values for true positive (tp), false positive (fp), false negative (fn), and true negative (tn)
fp = df_tasks['useless_steps'].sum()
fn = df_tasks['actions_missing'].sum()
tp = df_tasks['detected_actions'].sum() - fp - df_tasks['matched_wrong_pattern'].sum() - df_tasks['nlp_error'].sum()
tn = sentences - fp - fn - tp

# Calculate the accuracy, precision, recall, and f1 score
total = tp + fn + tn + fp
accuracy = (tp + tn) / total * 100
precision = tp / (tp + fp) * 100
recall = tp / (tp + fn) * 100
f1_score = (2 * precision * recall) / (precision + recall)

# Create a heatmap of the confusion matrix
confusion_matrix = [[tp, fp], [fn, tn]]
ax = sns.heatmap(confusion_matrix, cbar=False, annot=True, cmap='binary', fmt='.4g', alpha=0.7)

# Add text to the heatmap with the accuracy, precision, recall, and f1 score
stats = "\n\nAccuracy = {:0.2f}%        Precision = {:0.2f}%\nRecall = {:0.2f}%             F1 Score = {:0.2f}%".format(
    accuracy, precision, recall, f1_score)
ax.text(0.0, 2.5, stats, fontsize=17)

# Set the x and y labels and tick labels
ax.set_xlabel('Actual Actions\n', fontdict=dict(weight='bold'))
ax.set_ylabel('Predicted Actions\n', fontdict=dict(weight='bold'))
ax.xaxis.set_ticklabels(['Positive', 'Negative'])
ax.yaxis.set_ticklabels(['Positive', 'Negative'])
ax.xaxis.set_label_position('top')
ax.xaxis.tick_top()
ax.yaxis.set_label_position('left')
ax.yaxis.tick_left()

# Add labels to the cells of the heatmap
values = ["(TP)", "(FP)", "(FN)", "(TN)"]
ax.texts[0].set_text(ax.texts[0].get_text() + "\n" + values[0])
ax.texts[1].set_text(ax.texts[1].get_text() + "\n" + values[1])
ax.texts[2].set_text(ax.texts[2].get_text() + "\n" + values[2])
ax.texts[3].set_text(ax.texts[3].get_text() + "\n" + values[3])

# Save the plot as a PDF file
plt.savefig('./charts/confusion_matrix_total.pdf', dpi=200, bbox_inches="tight")

```


    
![png](output_22_0.png)
    

