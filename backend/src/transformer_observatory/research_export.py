import csv
import io


def report_csv(report):
    result = report["result"]
    if report["kind"] == "intervention":
        rows = result["candidates"]
    elif report["kind"] == "statistics":
        rows = result["rows"]
    else:
        rows = [
            dict(position=t, token_id=token["id"], stage=name, x=xyz[0], y=xyz[1], z=xyz[2])
            for t, token in enumerate(result["tokens"])
            for name, xyz in zip(result["stages"], result["coordinates"][t])
        ]
    output = io.StringIO(newline="")
    if rows:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            safe = {
                key: "'" + value
                if isinstance(value, str) and value.startswith(("=", "+", "-", "@", "\t", "\r"))
                else value
                for key, value in row.items()
            }
            writer.writerow(safe)
    return output.getvalue()
