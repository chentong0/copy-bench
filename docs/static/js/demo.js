
function DropDownTrigger({text}) {
    return (
        <button className="button" aria-haspopup="true" aria-controls="dropdown-menu">
            <span className="is-clipped">{text}</span>
            <span className="icon is-small"><i className="fas fa-angle-down" aria-hidden="true"></i></span>
        </button>
    );
}

function DropDownMenuItem({text, active, onClick}) {
    let cls = "dropdown-item" + (active ? " is-active" : "");
    return (
        <a className={cls} onClick={onClick}>{text}</a>
    );
}
function DropDownMenu({list}) {
    return list.map((item, idx) => 
        <DropDownMenuItem key={idx} text={item.text} active={item.active} onClick={item.onClick} />
    );
}

function ButtonGroupItem({text, active, onClick}) {
    let cls = "button" + (active ? " is-selected is-success" : "");
    // let text_display = active ? "Top-" + text : text;
    let text_display = text;
    return (
        <button className={cls} onClick={onClick}>{text_display}</button>
    );
    // <button className="button is-success is-selected">Top-1</button>
}
function ButtonGroup({list}) {
    return list.map((item, idx) => 
        <ButtonGroupItem key={idx} text={item.text} active={item.active} onClick={item.onClick} />
    );
}

function Box({title, content}) {
    return (
        <div className="box">
            <h5>{title}</h5>
            <div dangerouslySetInnerHTML={{__html: content}}></div>
        </div>
    );
}

function AppDemo({data}) {
    const [tid, setTid] = React.useState(0);  // task id
    const [bid, setBid] = React.useState(0);  // book id

    function getDropdownMenuList(list, idValue, setIDFunc) {
        return list.map((item, idx) => {
            return {
                text: item,
                active: idx == idValue,
                onClick: () => setIDFunc(idx),
            }
        })
    }
    let task_list = [
        "Literal Copying",
        "Non-literal Copying",
        "Fact Recall",
    ]
    function toTitleCase(str) {
        return str.toLowerCase()
        .split(' ')
        .map((s) => s.charAt(0).toUpperCase() + s.substring(1))
        .join(' ');
    }
    function getTitle(data, tid, bid) {
        return toTitleCase(data[tid][bid]["title"])
    }
    function getPrefix(data, tid, bid) {
        return data[tid][bid]["input"];
    }
    function getReference(data, tid, bid) {
        // if data[tid][bid] has key "reference" return it, otherwise use key "reference_event"
        if ("reference" in data[tid][bid]) {
            return data[tid][bid]["reference"];
        }
        else {
            let r1 = data[tid][bid]["reference_events"].map(
                x => "<li>" + x + "</li>"
            ).join("");
            let r2 = data[tid][bid]["reference_characters"].map(
                x => "<li>" + x.join(", ") + "</li>"
            ).join("");
            return "<dev><h6>Events</h3><ul>" + r1 + "</ul><h6>Characters</h3><ul>" + r2 + "</ul></dev>";
        }
    }
    return (
        <div>
            <div className="columns">
                <div className="column">
                    {/* get centering */}
                    <div className="buttons has-addons is-left is-centered">
                        <ButtonGroup list={getDropdownMenuList(task_list, tid, setTid)} />
                    </div>
                </div>
            </div>
            <div className="columns is-centered">
                <div className="column is-half">
                    <div className="dropdown is-hoverable is-fullwidth">
                        <div className="dropdown-trigger">
                            <DropDownTrigger text={getTitle(data, tid, bid)} />
                        </div>
                        <div className="dropdown-menu" role="menu">
                            <div className="dropdown-content">
                                <DropDownMenu list={getDropdownMenuList(
                                    Array.from(Array(10).keys()).map(x => getTitle(data, tid, x)),
                                    bid, setBid
                                )} />
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            <Box title={["Prefix", "Prefix", "Question"][tid]} content={getPrefix(data, tid, bid)} />
            <Box title="Reference" content={getReference(data, tid, bid)} />
        </div>
    );
}


// function AppDemo() {
//     return <h1>Hello, world!</h1>;
// }

// console.log("Hello, world!");
// load /static/data/data.json to data waiting for rendering
let data = {};
// Promise.all([
//     fetch("https://raw.githubusercontent.com/chentong0/copy-bench/main/data/data.literal.json").then(resp => resp.json()),
//     fetch("https://raw.githubusercontent.com/chentong0/copy-bench/main/data/data.nonliteral.json").then(resp => resp.json()),
//     fetch("https://raw.githubusercontent.com/chentong0/copy-bench/main/data/data.qa.json").then(resp => resp.json()),
// ])
fetch("static/data/data.json")
    .then(resp => resp.json())
    .then(json_list => {
        // console.log("data loading");
        data = json_list;
        // console.log(data[0][0]);
        // console.log(data[1][0]);
        // console.log(data[2][0]);
        console.log("data loaded");
        for (let i = 0; i < 3; i++) {
            // iterate each item in data[i], if there "title" have already appear once, delete this item
            let title_set = new Set();
            data[i] = data[i].filter(item => {
                if (title_set.has(item["title"])) {
                    return false;
                }
                else {
                    title_set.add(item["title"]);
                    return true;
                }
            });
        }
    })
    .then(() => {
        const container = document.getElementById('app-demo');
        const root = ReactDOM.createRoot(container);
        root.render(<AppDemo data={data} />);
        console.log("rendered");
    })


// fetch("https://raw.githubusercontent.com/chentong0/copy-bench/main/data/data.literal.json")
//     .then(response => response.json())
//     .then(json => {
//         console.log("data loading");
//         data = json;
//         // console.log(data);
//         console.log(data[0]);
//         console.log("data loaded");
//     })
// console.log("Hello, world!");