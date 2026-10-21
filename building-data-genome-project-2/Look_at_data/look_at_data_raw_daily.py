from pathlib import Path
import tkinter as tk
from tkinter import ttk

import pandas as pd


data_folder = Path(__file__).parent / "data" / "meters" / "raw"
meter_files = sorted(path.stem for path in data_folder.glob("*.csv"))
data = pd.DataFrame()


window = tk.Tk()
window.title("Målerdata")
window.geometry("1300x700")

# Kontroller til venstre: målertype, søk, liste med målere og antall rader
controls = ttk.Frame(window, padding=10)
controls.pack(side=tk.LEFT, fill=tk.Y)

ttk.Label(controls, text="Målertype").pack(anchor=tk.W)
meter_type = tk.StringVar(value="chilledwater" if "chilledwater" in meter_files else meter_files[0])
meter_type_box = ttk.Combobox(controls, textvariable=meter_type, values=meter_files, state="readonly")
meter_type_box.pack(fill=tk.X, pady=(0, 10))

ttk.Label(controls, text="Søk etter måler").pack(anchor=tk.W)
search_text = tk.StringVar()
ttk.Entry(controls, textvariable=search_text).pack(fill=tk.X, pady=(0, 10))

only_with_data = tk.BooleanVar(value=True)
ttk.Checkbutton(controls, text="Bare målere med data", variable=only_with_data).pack(anchor=tk.W)

ttk.Label(controls, text="Målere (velg flere med Ctrl/Shift)").pack(anchor=tk.W, pady=(10, 0))
meter_list = tk.Listbox(controls, selectmode=tk.EXTENDED, width=40, height=25, exportselection=False)
meter_list.pack(fill=tk.Y, expand=True)

# Timeverdiene slås sammen til én verdi per dag
ttk.Label(controls, text="Slå sammen timer per dag med").pack(anchor=tk.W, pady=(10, 0))
aggregation = tk.StringVar(value="Sum")
ttk.Combobox(controls, textvariable=aggregation, values=["Sum", "Snitt"], state="readonly").pack(fill=tk.X)

row_frame = ttk.Frame(controls)
row_frame.pack(fill=tk.X, pady=10)
ttk.Label(row_frame, text="Antall rader").pack(side=tk.LEFT)
row_count = tk.IntVar(value=500)
ttk.Spinbox(row_frame, from_=10, to=20000, increment=100, textvariable=row_count, width=8).pack(side=tk.RIGHT)

show_button = ttk.Button(controls, text="Vis valgte målere")
show_button.pack(fill=tk.X)

# Tabell og statistikk til høyre
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

ttk.Label(right_side, text="Statistikk for valgte målere").pack(anchor=tk.W, pady=(10, 0))
stats_table = ttk.Treeview(right_side, show="headings", height=8)
stats_table.pack(fill=tk.X)

status = ttk.Label(right_side, text="")
status.pack(anchor=tk.W, pady=(5, 0))


def fill_table(tree, frame):
	tree.delete(*tree.get_children())
	columns = list(frame.columns)
	tree["columns"] = columns
	for column in columns:
		tree.heading(column, text=column)
		tree.column(column, width=160, anchor=tk.CENTER, stretch=False)
	for row in frame.itertuples(index=False, name=None):
		tree.insert("", tk.END, values=["" if pd.isna(value) else value for value in row])


def update_meter_list(*_):
	meters = data.columns[1:]
	if only_with_data.get():
		meters = [meter for meter in meters if data[meter].notna().any()]
	query = search_text.get().lower()
	meter_list.delete(0, tk.END)
	for meter in meters:
		if query in meter.lower():
			meter_list.insert(tk.END, meter)


def load_meter_type(*_):
	global data
	data = pd.read_csv(data_folder / f"{meter_type.get()}.csv", parse_dates=["timestamp"])
	window.title(f"Målerdata per dag - {meter_type.get()}.csv")
	update_meter_list()
	status.config(text=f"{meter_type.get()}: {len(data)} rader, {len(data.columns) - 1} målere")
	# Vis de fem første målerne som har data, så tabellen ikke starter tom
	for index in range(min(5, meter_list.size())):
		meter_list.selection_set(index)
	show_selected()


def show_selected():
	selected = [meter_list.get(index) for index in meter_list.curselection()]
	if not selected:
		status.config(text="Velg minst én måler i listen")
		return

	daily_data = to_daily(data.loc[:, ["timestamp", *selected]])
	fill_table(table, daily_data.iloc[: row_count.get()])

	stats = daily_data[selected].describe().round(2).T
	stats = stats.rename(columns={"count": "antall", "mean": "snitt", "std": "std", "min": "min", "max": "maks"})
	stats.insert(0, "måler", stats.index)
	fill_table(stats_table, stats)

	status.config(
		text=f"Viser {min(row_count.get(), len(daily_data))} av {len(daily_data)} dager for {len(selected)} målere"
	)


def to_daily(frame):
	grouped = frame.set_index("timestamp").resample("D")
	# min_count=1 gjør at dager uten målinger blir tomme i stedet for 0
	result = grouped.sum(min_count=1) if aggregation.get() == "Sum" else grouped.mean()
	result = result.round(2).reset_index()
	result["timestamp"] = result["timestamp"].dt.strftime("%Y-%m-%d")
	return result


meter_type_box.bind("<<ComboboxSelected>>", load_meter_type)
search_text.trace_add("write", update_meter_list)
only_with_data.trace_add("write", update_meter_list)
aggregation.trace_add("write", lambda *_: show_selected())
show_button.config(command=show_selected)
meter_list.bind("<Double-Button-1>", lambda _: show_selected())

load_meter_type()
window.mainloop()
