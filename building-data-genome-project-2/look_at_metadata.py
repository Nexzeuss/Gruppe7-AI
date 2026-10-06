from pathlib import Path
import tkinter as tk
from tkinter import ttk

import pandas as pd


file_path = Path(__file__).parent / "data" / "metadata" / "metadata.csv"
data = pd.read_csv(file_path)

meter_types = ["electricity", "chilledwater", "steam", "hotwater", "gas", "water", "irrigation", "solar"]
default_columns = [
	"building_id", "site_id", "primaryspaceusage", "sub_primaryspaceusage",
	"sqm", "yearbuilt", "numberoffloors", "timezone", "chilledwater",
]
ALL = "Alle"


window = tk.Tk()
window.title(f"Metadata - {file_path.name}")
window.geometry("1400x750")

# Kontroller til venstre: søk, filtre og hvilke kolonner som vises
controls = ttk.Frame(window, padding=10)
controls.pack(side=tk.LEFT, fill=tk.Y)

ttk.Label(controls, text="Søk etter bygning").pack(anchor=tk.W)
search_text = tk.StringVar()
ttk.Entry(controls, textvariable=search_text).pack(fill=tk.X, pady=(0, 10))


def add_filter(label, values):
	ttk.Label(controls, text=label).pack(anchor=tk.W)
	variable = tk.StringVar(value=ALL)
	ttk.Combobox(controls, textvariable=variable, values=[ALL, *values], state="readonly").pack(
		fill=tk.X, pady=(0, 10)
	)
	return variable


site_filter = add_filter("Site", sorted(data["site_id"].dropna().unique()))
usage_filter = add_filter("Bygningstype", sorted(data["primaryspaceusage"].dropna().unique()))
meter_filter = add_filter("Har måler", meter_types)

# Kolonnelisten viser hvor mange bygninger som har verdi i hver kolonne
ttk.Label(controls, text="Kolonner som vises (utfylt av totalt)").pack(anchor=tk.W)
column_list = tk.Listbox(controls, selectmode=tk.EXTENDED, width=40, height=20, exportselection=False)
column_list.pack(fill=tk.Y, expand=True)
for column in data.columns:
	column_list.insert(tk.END, f"{column}  ({data[column].notna().sum()}/{len(data)})")
	if column in default_columns:
		column_list.selection_set(tk.END)

show_button = ttk.Button(controls, text="Vis valgte kolonner")
show_button.pack(fill=tk.X, pady=(10, 0))

# Tabell og oppsummering til høyre
right_side = ttk.Frame(window, padding=10)
right_side.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

table_frame = ttk.Frame(right_side)
table_frame.pack(fill=tk.BOTH, expand=True)

table = ttk.Treeview(table_frame, show="headings")
vertical_scrollbar = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=table.yview)
horizontal_scrollbar = ttk.Scrollbar(table_frame, orient=tk.HORIZONTAL, command=table.xview)
table.configure(
	yscrollcommand=vertical_scrollbar.set,
	xscrollcommand=horizontal_scrollbar.set,
)
table.grid(row=0, column=0, sticky="nsew")
vertical_scrollbar.grid(row=0, column=1, sticky="ns")
horizontal_scrollbar.grid(row=1, column=0, sticky="ew")
table_frame.rowconfigure(0, weight=1)
table_frame.columnconfigure(0, weight=1)

summary_header = ttk.Frame(right_side)
summary_header.pack(fill=tk.X, pady=(10, 0))
ttk.Label(summary_header, text="Oppsummer kolonne").pack(side=tk.LEFT)
summary_column = tk.StringVar(value="primaryspaceusage")
ttk.Combobox(
	summary_header, textvariable=summary_column, values=list(data.columns), state="readonly", width=30
).pack(side=tk.LEFT, padx=10)

summary_frame = ttk.Frame(right_side)
summary_frame.pack(fill=tk.X)
summary_table = ttk.Treeview(summary_frame, show="headings", height=9)
summary_scrollbar = ttk.Scrollbar(summary_frame, orient=tk.VERTICAL, command=summary_table.yview)
summary_table.configure(yscrollcommand=summary_scrollbar.set)
summary_table.pack(side=tk.LEFT, fill=tk.X, expand=True)
summary_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

status = ttk.Label(right_side, text="")
status.pack(anchor=tk.W, pady=(5, 0))


def fill_table(tree, frame, width=140):
	tree.delete(*tree.get_children())
	columns = list(frame.columns)
	tree["columns"] = columns
	for column in columns:
		tree.heading(column, text=column)
		tree.column(column, width=width, anchor=tk.CENTER, stretch=False)
	for row in frame.itertuples(index=False, name=None):
		tree.insert("", tk.END, values=["" if pd.isna(value) else value for value in row])


def filtered_data():
	rows = data
	if site_filter.get() != ALL:
		rows = rows[rows["site_id"] == site_filter.get()]
	if usage_filter.get() != ALL:
		rows = rows[rows["primaryspaceusage"] == usage_filter.get()]
	if meter_filter.get() != ALL:
		rows = rows[rows[meter_filter.get()].notna()]
	query = search_text.get().lower()
	if query:
		rows = rows[rows["building_id"].str.lower().str.contains(query, regex=False)]
	return rows


def show_summary(rows):
	values = rows[summary_column.get()]
	if pd.api.types.is_numeric_dtype(values):
		# Tallkolonner: vanlig statistikk
		summary = values.describe().round(2).reset_index()
		summary.columns = ["mål", "verdi"]
	else:
		# Tekstkolonner: hvor mange bygninger som har hver verdi
		summary = values.fillna("(mangler)").value_counts().reset_index()
		summary.columns = ["verdi", "antall bygninger"]
		summary["andel %"] = (summary["antall bygninger"] / len(rows) * 100).round(1)
	fill_table(summary_table, summary, width=220)


def show_selected(*_):
	columns = [data.columns[index] for index in column_list.curselection()]
	if not columns:
		status.config(text="Velg minst én kolonne i listen")
		return

	rows = filtered_data()
	fill_table(table, rows[columns])
	show_summary(rows)
	status.config(text=f"Viser {len(rows)} av {len(data)} bygninger, {len(columns)} kolonner")


for variable in (search_text, site_filter, usage_filter, meter_filter, summary_column):
	variable.trace_add("write", show_selected)
show_button.config(command=show_selected)
column_list.bind("<Double-Button-1>", show_selected)

show_selected()
window.mainloop()
